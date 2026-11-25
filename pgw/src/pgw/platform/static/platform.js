// PGW platform: progressive enhancements only (every page works without JavaScript).
//   * copy buttons ([data-copy="#id"]), announced to screen readers
//   * theme toggle in the top bar and the theme choice in Settings
//   * filter selects that submit on change ([data-autosubmit])
(function () {
  "use strict";
  var KEY = "pgw-theme";
  var root = document.documentElement;

  var live = document.createElement("div");
  live.className = "sr-only";
  live.setAttribute("aria-live", "polite");
  document.body.appendChild(live);
  function announce(msg) {
    live.textContent = "";
    window.setTimeout(function () { live.textContent = msg; }, 30);
  }

  // ---- copy ------------------------------------------------------------------------------------
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.className = "sr-only";
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      if (ok) resolve(); else reject(new Error("copy failed"));
    });
  }
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.hidden = false;
    btn.addEventListener("click", function () {
      var el = document.querySelector(btn.getAttribute("data-copy"));
      if (!el) return;
      var text = "value" in el && el.value !== undefined ? el.value : el.textContent;
      var label = btn.querySelector(".copy-label");
      var before = label ? label.textContent : null;
      copyText(text || "").then(function () {
        btn.classList.add("is-copied");
        if (label) label.textContent = "Copied";
        announce("Copied to clipboard");
        window.setTimeout(function () {
          btn.classList.remove("is-copied");
          if (label) label.textContent = before;
        }, 1600);
      }, function () {
        var range = document.createRange();
        range.selectNodeContents(el);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        announce("Press Control C or Command C to copy");
      });
    });
  });

  // ---- theme -----------------------------------------------------------------------------------
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function stored() {
    try { var t = localStorage.getItem(KEY); return t === "light" || t === "dark" ? t : null; } catch (e) { return null; }
  }
  function effective() { return stored() || (media && media.matches ? "dark" : "light"); }
  function setTheme(choice) {
    try {
      if (choice === "system") localStorage.removeItem(KEY); else localStorage.setItem(KEY, choice);
    } catch (e) { /* this page only */ }
    if (choice === "system") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", choice);
    syncChoices();
  }
  function syncChoices() {
    var cur = stored() || "system";
    document.querySelectorAll("[data-theme-choice]").forEach(function (r) { r.checked = r.value === cur; });
    document.querySelectorAll(".theme-toggle").forEach(function (b) {
      b.setAttribute("aria-label", effective() === "dark" ? "Switch to light theme" : "Switch to dark theme");
    });
  }
  document.querySelectorAll(".theme-toggle").forEach(function (b) {
    b.hidden = false;
    b.addEventListener("click", function () { setTheme(effective() === "dark" ? "light" : "dark"); });
  });
  document.querySelectorAll("[data-theme-choice]").forEach(function (r) {
    r.addEventListener("change", function () { if (r.checked) setTheme(r.value); });
  });
  syncChoices();

  // ---- filters ---------------------------------------------------------------------------------
  document.querySelectorAll("select[data-autosubmit]").forEach(function (s) {
    s.addEventListener("change", function () { if (s.form) s.form.submit(); });
  });
})();
