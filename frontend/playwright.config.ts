import { defineConfig, devices } from '@playwright/test';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');

/** Switch targets via `--project=local` or `--project=production` (no duplicated specs). */
export const TARGETS = {
  local: {
    // Must match ALLOWED_ORIGINS (localhost, not 127.0.0.1) or browser API calls fail CORS.
    baseURL: 'http://localhost:5173',
    apiURL: 'http://localhost:8000',
    storageState: path.join(__dirname, 'e2e/.auth/user.local.json'),
  },
  production: {
    baseURL: 'https://guidaplate.vercel.app',
    apiURL: 'https://guidaplate-production.up.railway.app',
    storageState: path.join(__dirname, 'e2e/.auth/user.production.json'),
  },
} as const;

const runWebServer = process.env.E2E_TARGET !== 'production';

export default defineConfig({
  testDir: path.join(__dirname, 'e2e'),
  fullyParallel: false,
  workers: 1,
  timeout: 90_000,
  expect: { timeout: 20_000 },
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  outputDir: 'test-results',
  use: {
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    ...devices['Desktop Chrome'],
  },
  projects: [
    {
      name: 'setup-local',
      testMatch: /auth\.setup\.ts/,
      use: { baseURL: TARGETS.local.baseURL },
    },
    {
      name: 'local',
      dependencies: ['setup-local'],
      testMatch: /specs\/.*\.spec\.ts/,
      use: {
        baseURL: TARGETS.local.baseURL,
        storageState: TARGETS.local.storageState,
      },
    },
    {
      name: 'setup-production',
      testMatch: /auth\.setup\.ts/,
      use: { baseURL: TARGETS.production.baseURL },
    },
    {
      name: 'production',
      dependencies: ['setup-production'],
      testMatch: /specs\/.*\.spec\.ts/,
      use: {
        baseURL: TARGETS.production.baseURL,
        storageState: TARGETS.production.storageState,
      },
    },
  ],
  ...(runWebServer
    ? {
        webServer: [
          {
            command: `${path.join(REPO_ROOT, 'venv311/bin/python')} -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`,
            cwd: REPO_ROOT,
            url: `${TARGETS.local.apiURL}/api/health`,
            reuseExistingServer: true,
            timeout: 120_000,
          },
          {
            command: 'npm run dev -- --host localhost --port 5173',
            cwd: __dirname,
            url: TARGETS.local.baseURL,
            reuseExistingServer: true,
            timeout: 120_000,
          },
        ],
      }
    : {}),
});
