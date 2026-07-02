import { chromium } from 'playwright';

const logs = [];
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();

page.on('console', (msg) => {
  const text = msg.text();
  if (text.includes('fetchDailyBudget')) {
    logs.push(text);
  }
});

const email = `debug_${Date.now()}@example.com`;
const password = 'testpass123';

const reg = await fetch('http://localhost:8000/api/auth/register', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'Debug User',
    email,
    password,
    ckd_stage: 'G3a',
    weight_kg: 70,
  }),
});
if (!reg.ok) {
  console.error('Register failed', reg.status, await reg.text());
  process.exit(1);
}

await page.goto('http://localhost:5173/');
await page.waitForLoadState('networkidle');

await page.locator('input[type="email"]').fill(email);
await page.locator('input[type="password"]').fill(password);
await page.getByRole('button', { name: 'Log In' }).click();
await page.waitForTimeout(800);

await page.getByText('Meal Assessment', { exact: true }).click();
await page.waitForTimeout(2000);

const budgetVisible = await page.getByText("Today's Nutrient Budget").count();

console.log('--- CONSOLE LOGS (in order) ---');
for (const line of logs) {
  console.log(line);
}
console.log('--- DOM CHECK ---');
console.log("Today's Nutrient Budget visible?", budgetVisible > 0);

await browser.close();
