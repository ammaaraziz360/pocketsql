const question = document.querySelector('#question');
const runButton = document.querySelector('#run-query');
const output = document.querySelector('#sql-output');
const latency = document.querySelector('#latency');

const schema = `CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT, created_at TEXT);
CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, created_at TEXT, total_amount REAL, status TEXT);
CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, stock_quantity INTEGER, price REAL, category TEXT, created_at TEXT);`;

const developerSnippets = {
  quickstart: {
    path: 'terminal',
    code: `<span class="comment"># create an environment and install PocketSQL</span>\npython3.13 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -e '.[dev]'\n\n<span class="comment"># run the local playground</span>\npython website/server.py`
  },
  python: {
    path: 'inference.py',
    code: `<span class="comment"># generate one safe query from Python</span>\n<span class="kw">from</span> pocketsql.inference <span class="kw">import</span> load_model_from_checkpoint, generate_sql\n<span class="kw">from</span> pocketsql.model.tokenizer <span class="kw">import</span> load_tokenizer\n\ncheckpoint = <span class="orange">"checkpoints/base-semantic-v14-factorized-best-execution"</span>\ntokenizer = load_tokenizer(checkpoint)\nmodel = load_model_from_checkpoint(checkpoint, tokenizer)\nsql = generate_sql(model, schema, question, tokenizer)`
  },
  server: {
    path: 'server.py',
    code: `<span class="comment"># the included local server exposes:</span>\nPOST <span class="orange">/api/generate</span>\n\n{\n  <span class="orange">"question"</span>: <span class="orange">"How many orders this month?"</span>,\n  <span class="orange">"schema"</span>: <span class="orange">"CREATE TABLE orders (...)"</span>\n}\n\n<span class="comment"># override the checkpoint when starting it</span>\nPOCKETSQL_CHECKPOINT=checkpoints/my-model \\\n  python website/server.py`
  }
};

function escapeHtml(value) {
  return value.replace(/[&<>'"]/g, (character) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[character]));
}

function formatSql(sql) {
  const normalized = sql.trim().replace(/\s+/g, ' ')
    .replace(/\s*,\s*/g, ',\n  ')
    .replace(/\s+(FROM|JOIN|WHERE|GROUP BY|ORDER BY|LIMIT|HAVING|UNION)\s+/gi, '\n$1 ')
    .replace(/\s+(ON|AND|OR)\s+/gi, '\n  $1 ')
    .replace(/\s+(ASC|DESC)\s*$/gi, ' $1');
  let safe = escapeHtml(normalized);
  safe = safe.replace(/\b(SELECT|FROM|JOIN|WHERE|GROUP BY|ORDER BY|LIMIT|HAVING|UNION|ON|AND|OR|AS|ASC|DESC)\b/g, '<span class="kw">$1</span>');
  safe = safe.replace(/\b(SUM|COUNT|AVG|MIN|MAX)\b/g, '<span class="fn">$1</span>');
  safe = safe.replace(/(&#39;.*?&#39;)/g, '<span class="orange">$1</span>');
  safe = safe.replace(/\b(\d+)\b/g, '<span class="num">$1</span>');
  return safe;
}

async function generate() {
  if (!question.value.trim()) return;
  runButton.disabled = true;
  runButton.classList.add('is-loading');
  latency.textContent = 'RUNNING…';
  output.style.opacity = '.35';
  try {
    const response = await fetch('/api/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: question.value, schema})
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Inference failed');
    output.innerHTML = formatSql(payload.sql);
    latency.textContent = payload.latencyMs ? `${payload.latencyMs}MS` : 'LOCAL MODEL';
    document.querySelector('.result-status span:nth-child(2)').textContent = 'Valid read-only query';
  } catch (error) {
    output.textContent = error.message;
    latency.textContent = 'ERROR';
    document.querySelector('.result-status span:nth-child(2)').textContent = window.location.hostname.endsWith('github.io')
      ? 'Run locally for live MLX inference'
      : 'Model could not validate this query';
  } finally {
    output.style.opacity = '1';
    runButton.disabled = false;
    runButton.classList.remove('is-loading');
  }
}

document.querySelectorAll('.suggestions button').forEach((button) => {
  button.addEventListener('click', () => { question.value = button.dataset.question; generate(); });
});
runButton.addEventListener('click', generate);
question.addEventListener('keydown', (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') generate();
});

const codeBlock = document.querySelector('#code-block');
const codePath = document.querySelector('#code-path');
document.querySelectorAll('.dev-tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelector('.dev-tab.active').classList.remove('active');
    tab.classList.add('active');
    const snippet = developerSnippets[tab.dataset.tab];
    codeBlock.dataset.current = tab.dataset.tab;
    codeBlock.innerHTML = `<code>${snippet.code}</code>`;
    codePath.textContent = snippet.path;
  });
});

document.querySelector('#copy-code').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  const snippet = developerSnippets[codeBlock.dataset.current];
  const plainCode = snippet.code.replace(/<[^>]+>/g, '');
  await navigator.clipboard.writeText(plainCode);
  button.innerHTML = 'Copied <span>✓</span>';
  window.setTimeout(() => { button.innerHTML = 'Copy <span>□</span>'; }, 1600);
});
