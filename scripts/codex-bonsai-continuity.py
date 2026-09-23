"""Add scoped task continuity hooks to a local Codex invocation."""
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import stat
import select
import time
import subprocess
import sys
import tempfile
import tomllib


def toml_value(value):
    if isinstance(value, dict):
        return '{' + ','.join(json.dumps(k) + '=' + toml_value(v) for k, v in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ','.join(toml_value(v) for v in value) + ']'
    return json.dumps(value)


def inspect_hooks(binary, overrides, project):
    requests = [dict(id=1, method='initialize', params=dict(clientInfo=dict(name='pulse-continuity', version='1'), capabilities=dict(experimentalApi=True))),
                dict(method='initialized'), dict(id=2, method='hooks/list', params=dict(cwds=[str(project)]))]
    command = [binary, 'app-server']
    for key, value in overrides.items():
        command += ['-c', key + '=' + toml_value(value)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, cwd=project)
    buffered = b''
    deadline = time.monotonic() + 15
    def send(value):
        process.stdin.write((json.dumps(value) + '\n').encode()); process.stdin.flush()
    def receive(identifier):
        nonlocal buffered
        while time.monotonic() < deadline:
            while b'\n' in buffered:
                line, buffered = buffered.split(b'\n', 1)
                row = json.loads(line)
                if row.get('id') == identifier:
                    return row
            ready, _, _ = select.select([process.stdout], [], [], max(0, deadline-time.monotonic()))
            if not ready:
                break
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                break
            buffered += chunk
            if len(buffered) > 4*1024*1024:
                raise ValueError('Codex hook metadata exceeds the size limit.')
        raise ValueError('Codex did not return hook metadata. Codex 0.154 or compatible hook support is required.')
    try:
        send(requests[0]); receive(1)
        send(requests[1]); send(requests[2])
        row = receive(2)
        if 'error' in row:
            raise ValueError('Codex could not inspect continuity hooks.')
        entries = row['result']['data']
        if len(entries) != 1 or entries[0].get('errors'):
            raise ValueError('Codex reported a hook configuration error.')
        return entries[0]['hooks']
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        process.stdin.close(); process.stdout.close()


def register_trust(home, expected):
    """Add only absent exact hook hashes. Preserve existing config bytes."""
    home = Path(home)
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_fd = os.open(home / '.pulse-continuity-trust.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(lock_fd, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = home / 'config.toml'
        raw = b''
        mode = 0o600
        before = None
        if path.exists() or path.is_symlink():
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'rb') as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                    raise ValueError('Codex config must be a regular file owned by the current user.')
                mode = stat.S_IMODE(info.st_mode)
                before = (info.st_ino, info.st_mtime_ns, info.st_size)
                raw = stream.read(2 * 1024 * 1024 + 1)
            if len(raw) > 2 * 1024 * 1024:
                raise ValueError('Codex config exceeds the continuity registration limit.')
        states = tomllib.loads(raw.decode()).get('hooks', {}).get('state', {})
        additions = []
        for key, digest in expected.items():
            current = states.get(key)
            if current is not None:
                if current.get('trusted_hash') != digest:
                    raise ValueError('A continuity hook trust key already has a different hash. Review that exact hook with /hooks.')
                continue
            additions.append('\n# Pulse task continuity: exact hook definition only.\n[hooks.state.' + json.dumps(key) + ']\ntrusted_hash = ' + json.dumps(digest) + '\n')
        if not additions:
            return
        updated = raw + (b'\n' if raw and not raw.endswith(b'\n') else b'') + ''.join(additions).encode()
        tomllib.loads(updated.decode())
        fd, temporary = tempfile.mkstemp(prefix='.pulse-continuity-', dir=home)
        try:
            with os.fdopen(fd, 'wb') as stream:
                os.fchmod(stream.fileno(), mode)
                stream.write(updated); stream.flush(); os.fsync(stream.fileno())
            now = path.stat() if path.exists() else None
            actual = (now.st_ino, now.st_mtime_ns, now.st_size) if now else None
            if actual != before:
                raise ValueError('Codex config changed during hook registration. Retry the launch.')
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        print('Continuity registered exact hook hashes in ' + str(path) + '; existing settings are unchanged.', file=sys.stderr)


def configure(binary, settings, user_args, user_config):
    task = os.environ.get('BONSAI_CODEX_TASK')
    if not task or any(a in ('--help','-h','--version','-V') for a in user_args):
        return
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', task):
        raise ValueError('BONSAI_CODEX_TASK must be a stable task ID with letters, digits, underscores, or hyphens.')
    for i in range(1, len(user_config), 2):
        key = user_config[i].split('=', 1)[0].strip()
        parsed = tomllib.loads(key + '=0')
        if 'hooks' in parsed or any(name in parsed.get('features', {}) for name in ('hooks', 'codex_hooks')):
            raise ValueError('Continuity cannot combine caller hook overrides. Keep existing hooks in their normal config layers.')
    for i, arg in enumerate(user_args):
        if (arg in ('--enable', '--disable') and i+1 < len(user_args) and user_args[i+1] in ('hooks','codex_hooks')) or arg in ('--enable=hooks','--disable=hooks','--enable=codex_hooks','--disable=codex_hooks'):
            raise ValueError('Continuity requires its verified hook configuration.')
    project = Path(os.environ.get('BONSAI_CODEX_PROJECT', os.getcwd())).expanduser().resolve()
    for i, arg in enumerate(user_args):
        if arg in ('-C', '--cd') and i + 1 < len(user_args):
            project = Path(user_args[i + 1]).expanduser().resolve()
        elif arg.startswith('--cd='):
            project = Path(arg.split('=', 1)[1]).expanduser().resolve()
        elif arg.startswith('-C') and len(arg) > 2:
            project = Path(arg[2:]).expanduser().resolve()
    if not project.is_dir():
        raise ValueError('The continuity project must be an existing directory.')
    if '--worktree' in user_args:
        raise ValueError('Create the worktree first, then launch continuity from its directory.')
    helper = Path(__file__).resolve().with_name('local_context.py')
    if not helper.is_file():
        raise ValueError('The local continuity helper is missing.')
    vault = Path(os.environ.get('BONSAI_CODEX_VAULT', str(Path.home() / 'vault'))).expanduser().resolve()
    os.environ.setdefault('PULSE_CONTINUITY_JEV', '1')
    os.environ.update(PULSE_CONTINUITY_PROJECT=str(project), PULSE_CONTINUITY_TASK=task, PULSE_CONTINUITY_VAULT=str(vault))
    command = shlex.join([sys.executable, str(helper), 'hook']) + ' --project "$PULSE_CONTINUITY_PROJECT" --task "$PULSE_CONTINUITY_TASK" --vault-root "$PULSE_CONTINUITY_VAULT"'
    handler = dict(type='command', command=command, timeout=10, additionalContextLimit=5000)
    settings['features.hooks'] = True
    settings['hooks.SessionStart'] = [dict(matcher='^(startup|resume|clear|compact)$', hooks=[handler])]
    settings['hooks.UserPromptSubmit'] = [dict(hooks=[handler])]
    settings['hooks.PreCompact'] = [dict(matcher='^(manual|auto)$', hooks=[{k:v for k,v in handler.items() if k != 'additionalContextLimit'}])]
    hooks = inspect_hooks(binary, settings, project)
    ours = [h for h in hooks if h.get('source') == 'sessionFlags' and h.get('command') == command]
    if len(ours) != 3 or {h['eventName'] for h in ours} != {'sessionStart', 'preCompact', 'userPromptSubmit'} or any(not h['enabled'] for h in ours):
        raise ValueError('Codex did not enable all three exact continuity hooks. Check managed hook policy.')
    expected = {}
    for hook in ours:
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', hook['currentHash']):
            raise ValueError('Codex returned an unsupported hook hash.')
        expected[hook['key']] = hook['currentHash']
    register_trust(os.environ.get('CODEX_HOME', str(Path.home() / '.codex')), expected)
    checked = inspect_hooks(binary, settings, project)
    if any(not any(h['key'] == key and h['currentHash'] == digest and h['trustStatus'] in ('trusted','managed') for h in checked) for key,digest in expected.items()):
        raise ValueError('Codex did not confirm exact continuity hook trust.')
    init = subprocess.run([sys.executable, str(helper), 'init', '--project', str(project), '--task', task, '--objective', os.environ.get('BONSAI_CODEX_OBJECTIVE', 'Continue the current task. Replace this placeholder with the user objective before substantive work.')],
                          capture_output=True, text=True, timeout=10, check=True)
    state = json.loads(init.stdout)
    print('Continuity checkpoint: ' + state['path'], file=sys.stderr)
