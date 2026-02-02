const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const http = require('http');

let mainWindow;
let pythonProcess;
const PYTHON_PORT = 8001;
const PYTHON_HOST = '127.0.0.1';

// Function to find Python executable
function getPythonCommand() {
    const isWin = process.platform === 'win32';
    // Check for local venv first if it exists
    const venvPath = path.join(__dirname, '..', '.venv', isWin ? 'Scripts' : 'bin', isWin ? 'python.exe' : 'python3');
    // You might want to add logic to check if this file exists using fs.existsSync
    // For now, defaulting to system python if we can't be sure, or assuming user has it in path.
    // Actually, let's just use 'python3' and assume it's in PATH for Mac/Linux and 'python' for Windows
    return isWin ? 'python' : 'python3';
}

function startPythonBackend() {
    const pythonCmd = getPythonCommand();
    const scriptPath = path.join(__dirname, '..', 'src', 'server', 'main.py');

    // Set PYTHONPATH to include the project root so imports work
    const projectRoot = path.join(__dirname, '..');
    const env = {
        ...process.env,
        PYTHONPATH: projectRoot + (process.platform === 'win32' ? ';' : ':') + (process.env.PYTHONPATH || ''),
        PORT: PYTHON_PORT.toString()
    };

    console.log(`Starting Python backend: ${pythonCmd} ${scriptPath}`);

    pythonProcess = spawn(pythonCmd, [scriptPath], { env });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`[Python]: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`[Python API Error]: ${data}`);
    });

    pythonProcess.on('close', (code) => {
        console.log(`Python process exited with code ${code}`);
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
    // In development, you might want to load localhost:3000
    // HEADS UP: We are pointing to the OUT directory of web_app
    const indexPath = path.join(__dirname, '..', 'web_app', 'out', 'index.html');
    mainWindow.loadFile(indexPath).catch(err => console.log("Failed to load index.html. Did you run build:desktop?", err));

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
