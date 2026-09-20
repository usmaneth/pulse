import { PulseServer } from './server.js';

const port = Number(process.env.PORT || 8000);
const host = process.env.HOST || '0.0.0.0';

const server = new PulseServer({ port, host });

async function main() {
  await server.start();

  process.on('SIGINT', async () => {
    console.log('\nStopping Pulse server...');
    await server.stop();
    process.exit(0);
  });
}

main().catch(console.error);
