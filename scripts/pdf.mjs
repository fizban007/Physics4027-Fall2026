#!/usr/bin/env node
/*
 * Renders the lecture decks to PDF.
 *
 *   npm run pdf              every lecture
 *   npm run pdf 3 7          lectures 3 and 7, or by name: npm run pdf "Lecture 3"
 *
 * Each deck is served over a local HTTP server, opened in headless Chrome with
 * reveal's ?print-pdf mode plus the config overrides in PRESET.query, and sent
 * through the print preset in scripts/pdf-preset.js before printToPDF runs.
 * The preset is what makes the result stable: it waits for webfonts, images
 * and the MathJax queue, re-typesets the equations against the finished print
 * layout, and re-measures every page, so nothing is centred, shrunk or clipped
 * with numbers taken before the maths had its final size.
 *
 * Only Node's standard library and a locally installed Chrome or Chromium are
 * needed; the DevTools protocol is spoken over the global WebSocket client.
 */

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readdir, readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir, homedir } from 'node:os';
import { dirname, extname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const PRESET_SOURCE_PATH = join(ROOT, 'scripts', 'pdf-preset.js');

const PRESET = {
  // Output directory, relative to the repository root.
  outDir: 'pdf',
  // Passed to reveal as the query string, so they override each deck's own
  // Reveal.initialize() the same way they would in a browser.
  query: {
    'print-pdf': '',
    // One page per slide: show every fragment at once instead of one page per
    // click, which is reveal's default when printing.
    pdfSeparateFragments: false,
    showNotes: false,
    showSlideNumber: 'all'
  },
  // Re-typeset the equations after the print layout settles.
  reprocessMath: true,
  // Repeat each deck's .deck-footer line on every page.
  footer: true,
  // How long a deck may take to load, typeset and lay out.
  settleTimeoutMs: 180000,
  // How long one printToPDF call may take.
  printTimeoutMs: 120000
};

const BROWSER_CANDIDATES = [
  process.env.CHROME_PATH,
  process.env.PUPPETEER_EXECUTABLE_PATH,
  'google-chrome-stable',
  'google-chrome',
  'chromium',
  'chromium-browser',
  '/usr/bin/google-chrome-stable',
  '/usr/bin/chromium',
  join(homedir(), '.cache/ms-playwright/chromium-1140/chrome-linux/chrome'),
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium'
];

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json',
  '.map': 'application/json',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.avif': 'image/avif',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.pdf': 'application/pdf',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.eot': 'application/vnd.ms-fontobject',
  '.txt': 'text/plain; charset=utf-8'
};

/* ------------------------------------------------------------------ server */

function startServer(root) {
  const server = createServer(async (req, res) => {
    const path = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
    const target = join(root, path.endsWith('/') ? join(path, 'index.html') : path);
    if (!target.startsWith(root + sep)) {
      res.writeHead(403).end('forbidden');
      return;
    }
    try {
      const body = await readFile(target);
      res.writeHead(200, {
        'content-type': MIME[extname(target).toLowerCase()] || 'application/octet-stream',
        // Decks change between runs; never serve a stale slide or figure.
        'cache-control': 'no-store'
      });
      res.end(body);
    } catch {
      res.writeHead(404).end('not found: ' + path);
    }
  });
  return new Promise((done) => {
    server.listen(0, '127.0.0.1', () => {
      done({ server, origin: `http://127.0.0.1:${server.address().port}` });
    });
  });
}

/* -------------------------------------------------------------- devtools */

class DevTools {
  constructor(socket) {
    this.socket = socket;
    this.nextId = 1;
    this.pending = new Map();
    this.listeners = new Map();
    socket.addEventListener('message', (event) => this.onMessage(event.data));
    socket.addEventListener('error', () => this.failAll(new Error('DevTools socket error')));
    socket.addEventListener('close', () => this.failAll(new Error('DevTools socket closed')));
  }

  static async connect(url) {
    const socket = new WebSocket(url);
    await new Promise((ok, bad) => {
      socket.addEventListener('open', ok, { once: true });
      socket.addEventListener('error', () => bad(new Error('could not open ' + url)), { once: true });
    });
    return new DevTools(socket);
  }

  onMessage(raw) {
    let message;
    try {
      message = JSON.parse(raw);
    } catch {
      return;
    }
    if (message.id && this.pending.has(message.id)) {
      const { ok, bad } = this.pending.get(message.id);
      this.pending.delete(message.id);
      if (message.error) bad(new Error(`${message.id}: ${message.error.message}`));
      else ok(message.result);
      return;
    }
    for (const listener of this.listeners.get(message.method) || []) {
      try {
        listener(message.params);
      } catch { /* diagnostics must never break a render */ }
    }
  }

  on(method, listener) {
    if (!this.listeners.has(method)) this.listeners.set(method, new Set());
    this.listeners.get(method).add(listener);
  }

  once(method, ms, label) {
    const waiter = new Promise((ok) => {
      const listener = (params) => {
        this.listeners.get(method).delete(listener);
        ok(params);
      };
      this.on(method, listener);
    });
    return withTimeout(waiter, ms, label);
  }

  failAll(error) {
    for (const { bad } of this.pending.values()) bad(error);
    this.pending.clear();
  }

  send(method, params = {}, sessionId) {
    const id = this.nextId++;
    const payload = { id, method, params };
    if (sessionId) payload.sessionId = sessionId;
    this.socket.send(JSON.stringify(payload));
    return new Promise((ok, bad) => this.pending.set(id, { ok, bad }));
  }

  close() {
    this.socket.close();
  }
}

function withTimeout(promise, ms, label) {
  let timer;
  const timeout = new Promise((_, bad) => {
    timer = setTimeout(() => bad(new Error(`timed out after ${ms}ms waiting for ${label}`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/* ----------------------------------------------------------------- chrome */

async function launchChrome(executable, extraArgs = []) {
  const profile = await mkdtemp(join(tmpdir(), 'phys4027-pdf-'));
  const args = [
    '--headless',
    '--disable-gpu',
    '--disable-dev-shm-usage',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-component-extensions-with-background-pages',
    '--remote-debugging-port=0',
    '--remote-allow-origins=*',
    '--hide-scrollbars',
    '--force-device-scale-factor=1',
    // Larger than any preset page, so the deck lays its canvas out at full
    // size while loading. printToPDF sets the PDF geometry.
    '--window-size=1600,1200',
    `--user-data-dir=${profile}`,
    ...extraArgs,
    'about:blank'
  ];
  const proc = spawn(executable, args, { stdio: ['ignore', 'ignore', 'pipe'] });
  const stderr = [];
  proc.stderr.setEncoding('utf8');
  proc.stderr.on('data', (chunk) => stderr.push(chunk));

  const portFile = join(profile, 'DevToolsActivePort');
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) break;
    try {
      const port = parseInt((await readFile(portFile, 'utf8')).split('\n')[0], 10);
      if (port) {
        const version = await (await fetch(`http://127.0.0.1:${port}/json/version`)).json();
        return {
          proc,
          profile,
          browser: version.Browser,
          webSocketDebuggerUrl: version.webSocketDebuggerUrl
        };
      }
    } catch { /* the port file appears once Chrome is listening */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  proc.kill('SIGKILL');
  await rm(profile, { recursive: true, force: true });
  throw new Error(
    `${executable} did not start a DevTools server\n` +
    stderr.join('').split('\n').filter(Boolean).slice(-6).join('\n')
  );
}

function findBrowser(explicit) {
  const candidates = explicit ? [explicit] : BROWSER_CANDIDATES.filter(Boolean);
  const pathDirs = (process.env.PATH || '').split(sep === '\\' ? ';' : ':');
  for (const candidate of candidates) {
    if (candidate.includes('/') || candidate.includes('\\')) {
      if (existsSync(candidate)) return candidate;
      continue;
    }
    for (const dir of pathDirs) {
      const hit = join(dir, candidate);
      if (existsSync(hit)) return hit;
    }
  }
  throw new Error(
    'no Chrome or Chromium found. Set CHROME_PATH, or install google-chrome / chromium.'
  );
}

/* ------------------------------------------------------------------ decks */

async function findDecks(root) {
  const entries = await readdir(root, { withFileTypes: true });
  const decks = [];
  for (const entry of entries) {
    const match = /^Lecture\s+(\d+)$/i.exec(entry.name);
    if (!entry.isDirectory() || !match) continue;
    if (!existsSync(join(root, entry.name, 'index.html'))) continue;
    decks.push({ dir: entry.name, number: Number(match[1]) });
  }
  return decks.sort((a, b) => a.number - b.number);
}

function selectDecks(decks, requested) {
  if (requested.length === 0) return decks;
  const chosen = [];
  for (const term of requested) {
    const numeric = /^\d+$/.test(term);
    const wanted = term.toLowerCase();
    // A bare number is a lecture number; a name matches exactly first and by
    // substring after that.
    let hits = decks.filter((deck) =>
      numeric ? String(deck.number) === term : deck.dir.toLowerCase() === wanted
    );
    if (hits.length === 0 && !numeric) {
      hits = decks.filter((deck) => deck.dir.toLowerCase().includes(wanted));
    }
    if (hits.length === 0) throw new Error(`no lecture matching "${term}"`);
    for (const deck of hits) if (!chosen.includes(deck)) chosen.push(deck);
  }
  return chosen.sort((a, b) => a.number - b.number);
}

const outputFile = (deck, outDir) =>
  join(outDir, `lecture-${String(deck.number).padStart(2, '0')}.pdf`);

// Keep log lines short for the usual case of writing inside the repository.
const displayPath = (file) => {
  const rel = relative(ROOT, file);
  return rel.startsWith('..') ? file : rel;
};

/* ----------------------------------------------------------------- render */

async function renderDeck({ deck, origin, outDir, cdp, sessionId, options }) {
  const send = (method, params) => cdp.send(method, params, sessionId);
  const evaluate = async (expression, ms, label) => {
    const result = await withTimeout(
      cdp.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }, sessionId),
      ms, label
    );
    if (result.exceptionDetails) {
      const details = result.exceptionDetails;
      throw new Error(
        details.exception?.description || details.text || 'page evaluation failed'
      );
    }
    return result.result.value;
  };

  const url = `${origin}/${encodeURIComponent(deck.dir)}/?` +
    Object.entries(PRESET.query)
      .map(([key, value]) => (value === '' ? key : `${key}=${value}`))
      .join('&');

  // The load event also guarantees the page's JavaScript context exists
  // before the preset is called.
  const loaded = cdp.once('Page.loadEventFired', PRESET.settleTimeoutMs, `${deck.dir} to load`);
  const navigation = await send('Page.navigate', { url });
  if (navigation.errorText) throw new Error(`navigation failed: ${navigation.errorText}`);
  await loaded;

  const settleArgs = JSON.stringify({
    reprocessMath: options.reprocessMath,
    timeoutMs: PRESET.settleTimeoutMs
  });
  const layout = await evaluate(
    `(async () => {
       const settled = await PhysPdf.settle(${settleArgs});
       const page = await PhysPdf.paginate(${JSON.stringify({
         footer: options.footer
       })});
       return { settled, page };
     })()`,
    PRESET.settleTimeoutMs + 30000,
    `${deck.dir} to finish typesetting`
  );

  const { settled, page } = layout;
  const pdf = await withTimeout(
    send('Page.printToPDF', {
      printBackground: true,
      preferCSSPageSize: false,
      // reveal sizes its print pages in CSS pixels; 96 of them make an inch.
      paperWidth: settled.pageWidthPx / 96,
      paperHeight: settled.pageHeightPx / 96,
      marginTop: 0,
      marginBottom: 0,
      marginLeft: 0,
      marginRight: 0,
      scale: 1,
      displayHeaderFooter: false
    }),
    PRESET.printTimeoutMs,
    `${deck.dir} to print`
  );

  const file = outputFile(deck, outDir);
  const bytes = Buffer.from(pdf.data, 'base64');
  await writeFile(file, bytes);

  return {
    file,
    bytes: bytes.length,
    pages: page.pages,
    url,
    title: settled.title,
    math: settled.math,
    images: settled.images,
    shrunk: page.shrunk
  };
}

/* ------------------------------------------------------------------- main */

function parseArgs(argv) {
  const options = {
    outDir: PRESET.outDir,
    browser: null,
    reprocessMath: PRESET.reprocessMath,
    footer: PRESET.footer,
    requested: [],
    help: false
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--help' || arg === '-h') options.help = true;
    else if (arg === '--out') options.outDir = argv[++i] || PRESET.outDir;
    else if (arg === '--browser') options.browser = argv[++i];
    else if (arg === '--no-math-reprocess') options.reprocessMath = false;
    else if (arg === '--no-footer') options.footer = false;
    else if (arg.startsWith('-')) throw new Error(`unknown option ${arg}`);
    else options.requested.push(arg);
  }
  return options;
}

const HELP = `Usage: npm run pdf [--] [lecture ...] [options]

Renders lecture decks to PDF with the print preset in scripts/pdf-preset.js.

  lecture ...        lecture numbers or directory names (default: all)

Options:
  --out <dir>        output directory (default: ${PRESET.outDir})
  --browser <path>   Chrome or Chromium executable (default: found in PATH,
                     or the CHROME_PATH environment variable)
  --no-math-reprocess
                     skip the second MathJax pass over the finished layout
  --no-footer        do not repeat the deck footer on every page
`;

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(HELP);
    return 0;
  }

  const presetSource = await readFile(PRESET_SOURCE_PATH, 'utf8');
  const decks = selectDecks(await findDecks(ROOT), options.requested);
  if (decks.length === 0) throw new Error(`no "Lecture N" directories under ${ROOT}`);

  const outDir = isAbsolute(options.outDir)
    ? options.outDir
    : resolve(ROOT, options.outDir);
  await mkdir(outDir, { recursive: true });

  const { server, origin } = await startServer(ROOT);
  const executable = findBrowser(options.browser);
  let chrome;
  let cdp;
  const failures = [];
  const started = Date.now();

  process.stdout.write(`Rendering ${decks.length} deck${decks.length > 1 ? 's' : ''} to ${outDir}\n`);

  try {
    try {
      chrome = await launchChrome(executable);
    } catch (error) {
      process.stdout.write(`${error.message}\retrying with --no-sandbox\n`);
      chrome = await launchChrome(executable, ['--no-sandbox']);
    }
    process.stdout.write(`using ${chrome.browser}\n`);

    cdp = await DevTools.connect(chrome.webSocketDebuggerUrl);
    cdp.on('Runtime.exceptionThrown', (params) => {
      const text = params.exceptionDetails?.exception?.description || params.exceptionDetails?.text;
      if (text) process.stdout.write(`  page error: ${text.split('\n')[0]}\n`);
    });

    const { targetId } = await cdp.send('Target.createTarget', { url: 'about:blank' });
    const { sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true });
    const send = (method, params) => cdp.send(method, params, sessionId);

    await send('Page.enable');
    await send('Runtime.enable');
    await send('Page.addScriptToEvaluateOnNewDocument', { source: presetSource });

    for (const deck of decks) {
      const label = `${deck.dir} (Lecture ${deck.number})`;
      try {
        const result = await renderDeck({ deck, origin, outDir, cdp, sessionId, options });
        const pages = `${result.pages} page${result.pages > 1 ? 's' : ''}`;
        const size = `${(result.bytes / 1024 / 1024).toFixed(1)} MB`;
        const target = displayPath(result.file);
        process.stdout.write(`  ${label}: ${pages}, ${size} -> ${target}\n`);
        for (const slide of result.shrunk) {
          process.stdout.write(
            `    warning: slide ${slide.page} "${slide.heading}" measures ` +
            `${slide.contentWidth}x${slide.contentHeight}px, more than its ` +
            `${slide.canvasWidth}x${slide.canvasHeight}px canvas; printed at ` +
            `${slide.percent}%\n`
          );
        }
      } catch (error) {
        failures.push({ deck, message: error.message });
        process.stdout.write(`  ${label}: FAILED - ${error.message}\n`);
      }
    }
  } finally {
    if (cdp) {
      try { await cdp.send('Browser.close'); } catch { /* already closing */ }
      cdp.close();
    }
    if (chrome) {
      chrome.proc.kill('SIGKILL');
      await rm(chrome.profile, { recursive: true, force: true });
    }
    server.close();
  }

  const seconds = ((Date.now() - started) / 1000).toFixed(1);
  if (failures.length) {
    process.stdout.write(`\n${failures.length} of ${decks.length} deck(s) failed after ${seconds}s\n`);
    return 1;
  }
  process.stdout.write(`done in ${seconds}s\n`);
  return 0;
}

main()
  .then((code) => process.exit(code))
  .catch((error) => {
    process.stderr.write(`pdf: ${error.message}\n`);
    process.exitCode = 1;
  });
