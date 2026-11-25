// Apply the stored theme before first paint (loaded synchronously in <head>; no inline script under the CSP).
(function () {
  try {
    var t = localStorage.getItem("pgw-theme");
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  } catch (e) {
    /* storage unavailable: follow the system theme */
  }
  document.documentElement.classList.add("js");
})();
