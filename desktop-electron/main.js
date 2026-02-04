const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');
const fs = require('fs');
const os = require('os');

const logFile = path.join(os.homedir(), 'investa_app.log');

function logToFile(message) {
    const timestamp = new Date().toISOString();
    fs.appendFileSync(logFile, `[${timestamp}] ${message}\n`);
}

logToFile('--- App Starting ---');

let mainWindow;
let pythonProcess;
const PYTHON_PORT = 8001;
const PYTHON_HOST = '127.0.0.1';

// Function to find Python executable
function getPythonCommand() {
    const isWin = process.platform === 'win32';
    // If packaged, we might want to bundle a python env, but for now we assume system python/venv
    return isWin ? 'python' : 'python3';
}

function startPythonBackend() {
    const isDev = require('electron-is-dev');
    const pythonCmd = getPythonCommand();

    // In production, backend files are in extraResources (Resources folder)
    const baseDir = isDev ? path.join(__dirname, '..') : process.resourcesPath;

    const scriptPath = path.join(baseDir, 'src', 'server', 'main.py');

    // Set PYTHONPATH to include the project root so imports work
    const projectRoot = baseDir;
    const env = {
        ...process.env,
        PYTHONPATH: projectRoot + (process.platform === 'win32' ? ';' : ':') + (process.env.PYTHONPATH || ''),
        PORT: PYTHON_PORT.toString()
    };

    logToFile(`Starting Python backend: ${pythonCmd} ${scriptPath}`);
    logToFile(`PYTHONPATH: ${env.PYTHONPATH}`);

    pythonProcess = spawn(pythonCmd, [scriptPath], { env });

    pythonProcess.stdout.on('data', (data) => {
        logToFile(`[Python]: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        logToFile(`[Python API Error]: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        logToFile(`Python process exited with code ${code}`);
    });
}

function checkServerReady() {
    return new Promise((resolve, reject) => {
        const tryConnect = (retries = 20) => {
            http.get(`http://${PYTHON_HOST}:${PYTHON_PORT}/`, (res) => {
                if (res.statusCode === 200) {
                    resolve();
                } else {
                    if (retries > 0) setTimeout(() => tryConnect(retries - 1), 1000);
                    else reject(new Error('Server returned non-200 status'));
                }
            }).on('error', (err) => {
                if (retries > 0) setTimeout(() => tryConnect(retries - 1), 1000);
                else reject(err);
            });
        };
        tryConnect();
    });
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1280,
        height: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            // preload: path.join(__dirname, 'preload.js'),
        },
    });

    // Open DevTools for debugging
    // mainWindow.webContents.openDevTools();

    // Load the Next.js static export
    const isDev = require('electron-is-dev');
    const baseDir = isDev ? path.join(__dirname, '..') : app.getAppPath();
    const indexPath = path.join(baseDir, 'web_app', 'out', 'index.html');
    mainWindow.loadFile(indexPath).catch(err => console.error("Failed to load index.html. Path tried: " + indexPath, err));

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
    app.quit();
} else {
    app.on('second-instance', (event, commandLine, workingDirectory) => {
        if (mainWindow) {
            if (mainWindow.isMinimized()) mainWindow.restore();
            mainWindow.focus();
        }
    });

    app.whenReady().then(async () => {
        startPythonBackend();

        try {
            console.log('Waiting for backend to be ready...');
            await checkServerReady();
            console.log('Backend ready, creating window...');
            createWindow();
        } catch (err) {
            console.error('Failed to start backend:', err);
            // Create window anyway to show error?
            createWindow();
        }

        app.on('activate', () => {
            if (mainWindow === null) createWindow();
        });
    });

    app.on('window-all-closed', () => {
        app.quit();
    });

    app.on('will-quit', () => {
        if (pythonProcess) {
            pythonProcess.kill();
        }
    });
}
