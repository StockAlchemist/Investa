/* Serve the production build: `npm run start -- [-H host] [-p port]`.
 * next.config.ts sets `output: "standalone"`, which `next start` does not
 * serve. The standalone server omits the static assets and public/, so copy
 * them beside it first (the layout the Dockerfile builds).
 */
/* eslint-disable @typescript-eslint/no-require-imports -- Standalone CommonJS launcher. */
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const distDir = process.env.INVESTA_NEXT_DIST_DIR || '.next';
const standalone = path.join(root, distDir, 'standalone');
const server = path.join(standalone, 'server.js');

if (!fs.existsSync(server)) {
    console.error(`No standalone build at ${server}. Run \`npm run build\` first.`);
    process.exit(1);
}

const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
    if (args[i] === '-H' || args[i] === '--hostname') process.env.HOSTNAME = args[++i];
    else if (args[i] === '-p' || args[i] === '--port') process.env.PORT = args[++i];
}

for (const [from, to] of [
    [path.join(root, distDir, 'static'), path.join(standalone, distDir, 'static')],
    [path.join(root, 'public'), path.join(standalone, 'public')],
]) {
    if (fs.existsSync(from)) {
        fs.rmSync(to, { recursive: true, force: true }); // drop hashed assets of older builds
        fs.cpSync(from, to, { recursive: true });
    }
}

require(server);
