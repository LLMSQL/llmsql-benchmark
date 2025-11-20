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
