// Entry point for the Pulse gateway. See docs/QWEN38.md.
//
//   node dist/gateway/index.js [--config path/to/gateway.json]

import { loadConfig } from './config.js';
import { Gateway } from './server.js';
import { log } from './log.js';

async function main(): Promise<void> {
  const config = loadConfig();
  const gateway = new Gateway(config);
  await gateway.start();

  let stopping = false;
  const shutdown = async (signal: string) => {
    if (stopping) {
      log('warn', 'second signal; exiting now', { signal });
      process.exit(1);
    }
    stopping = true;
    log('info', 'shutting down', { signal, grace_ms: config.shutdownGraceMs });
    try {
      await gateway.stop();
    } catch (error) {
      log('error', 'error during shutdown', { error: String(error) });
    }
    log('info', 'stopped');
    process.exit(0);
  };
  process.on('SIGINT', () => void shutdown('SIGINT'));
  process.on('SIGTERM', () => void shutdown('SIGTERM'));
}

main().catch((error) => {
  log('error', 'gateway failed to start', { error: String(error) });
  process.exit(1);
});
