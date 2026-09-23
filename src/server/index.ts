import { PulseServer } from './server.js';

const port = Number(process.env.PORT || 8000);
const host = process.env.HOST || '0.0.0.0';

const server = new PulseServer({ port, host, backendUrl: process.env.PULSE_BACKEND_URL });

async function main() {
  await server.start();

  // SIGTERM matters as much as SIGINT: it is what systemd, docker and
  // kubernetes send first. Without a handler the process is killed outright
  // after the grace period, dropping in-flight requests.
  const shutdown = async (signal: string) => {
    console.log(`\nReceived ${signal}, stopping Pulse server...`);
    try {
      await server.stop();
    } catch (err) {
      console.error('error during shutdown:', err);
    }
    process.exit(0);
  };
  process.on('SIGINT', () => void shutdown('SIGINT'));
  process.on('SIGTERM', () => void shutdown('SIGTERM'));
}

main().catch(console.error);
