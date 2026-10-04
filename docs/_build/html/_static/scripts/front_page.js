// === LLMSQL Front Page Search Script ===

const searchInput = document.querySelector('.sidebar-search');
let highlights = [];
let currentIndex = 0;

function removeHighlights() {
  document.querySelectorAll('.search-highlight').forEach(span => {
    const parent = span.parentNode;
    parent.replaceChild(document.createTextNode(span.textContent), span);
    parent.normalize();
  });
  highlights = [];
}

function highlightMatches(query) {
  if (!query) return;
  const contentElements = document.querySelectorAll('main p, main li, main td, main pre, main h1, main h2, main h3');
  const regex = new RegExp(`(${query})`, 'gi');
  contentElements.forEach(el => {
    el.innerHTML = el.textContent.replace(regex, '<span class="search-highlight">$1</span>');
  });
  highlights = [...document.querySelectorAll('.search-highlight')];
}

function scrollToHighlight(index) {
  if (highlights.length === 0) return;
  highlights.forEach(h => h.style.background = '#fff89a');
  const el = highlights[index];
  el.style.background = '#ffe666';
  el.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

if (searchInput) {
  searchInput.addEventListener('input', e => {
    removeHighlights();
    const query = e.target.value.trim();
    if (query) highlightMatches(query);
  });

  searchInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (highlights.length > 0) {
        scrollToHighlight(currentIndex);
        currentIndex = (currentIndex + 1) % highlights.length;
      }
    }
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  const container = document.getElementById('leaderboard-container');
  if (!container) return;

  try {
    const response = await fetch('_static/leaderboard.json');
    const rows = await response.json();
    renderLeaderboard(rows);
  } catch (e) {
    container.innerHTML = '<p>Error loading leaderboard 😢</p>';
    console.error(e);
  }
});

function renderLeaderboard(rows) {
  const container = document.getElementById('leaderboard-container');
  container.innerHTML = '';

  const table = document.createElement('table');
  table.className = 'leaderboard-table';

  const thead = document.createElement('thead');
  thead.innerHTML = `
    <tr>
      <th>Rank</th>
      <th>Model</th>
      <th>Type</th>
      <th>Fewshots</th>
      <th>Backend</th>
      <th>Accuracy</th>
      <th>Date</th>
    </tr>`;
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  rows.forEach((row, i) => {
    const tr = document.createElement('tr');

    // Берём только вторую часть после слеша
    const modelName = row.model.includes('/') ? row.model.split('/')[1] : row.model;

    // Модель с ссылкой
    const modelCell = document.createElement('td');
    if (row.url) {
      const a = document.createElement('a');
      a.href = row.url;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = modelName;   // <-- здесь только вторая часть
      modelCell.appendChild(a);
    } else {
      modelCell.textContent = modelName;
    }

    // Accuracy
    const accuracyCell = document.createElement('td');
    const barContainer = document.createElement('div');
    barContainer.className = 'accuracy-bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.width = `${(row.accuracy*100).toFixed(2)}%`;
    const text = document.createElement('span');
    text.textContent = `${(row.accuracy*100).toFixed(2)}%`;
    barContainer.appendChild(fill);
    barContainer.appendChild(text);
    accuracyCell.appendChild(barContainer);

    // Вставка остальных ячеек
    tr.innerHTML += `<td>${i+1}</td>`;
    tr.appendChild(modelCell);
    tr.innerHTML += `
      <td>${row.type}</td>
      <td>${row.fewshots}</td>
      <td>${row.backend}</td>
    `;
    tr.appendChild(accuracyCell);
    tr.innerHTML += `<td>${row.date}</td>`;

    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  container.appendChild(table);
}