(function () {
  "use strict";

  var VIEW_KEY = "uniwiseViewMode_v1";

  /* ── Detect real device (same logic as view-mode-init.js) ── */
  function isRealMobile() {
    var ua = navigator.userAgent || "";
    var uaMatch = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini|Mobile|Tablet/i.test(ua);
    var narrow = window.screen.width <= 960;
    var touch  = ("ontouchstart" in window) || (navigator.maxTouchPoints > 0);
    return uaMatch || (narrow && touch);
  }

  var realMobile = isRealMobile();

  /* ════════════════════════════════════════════════════
  ════════════════════════════════════════════════════ */
  function setVh() {
    document.documentElement.style.setProperty("--vh", (window.innerHeight * 0.01) + "px");
  }
  setVh();
  window.addEventListener("resize", setVh, { passive: true });
  window.addEventListener("orientationchange", function () { setTimeout(setVh, 150); }, { passive: true });

  document.addEventListener("DOMContentLoaded", function () {

    /* ── POV TOGGLE (desktop only) ─────────────── */
    var povBtn  = document.getElementById("povToggleBtn");
    var povIcon = document.getElementById("povToggleIcon");

    if (povBtn) {
      /* Hide the POV button entirely on real mobile — useless there */
      if (realMobile) {
        povBtn.style.display = "none";
      } else {
        var isMobile = localStorage.getItem(VIEW_KEY) === "mobile";

        function applyPov(mobile) {
          if (mobile) {
            document.body.classList.add("pov-phone-frame");
            povBtn.classList.add("is-mobile");
            if (povIcon) { povIcon.classList.remove("bi-laptop"); povIcon.classList.add("bi-phone"); }
            povBtn.setAttribute("title", "Switch to Desktop view");
          } else {
            document.body.classList.remove("pov-phone-frame");
            povBtn.classList.remove("is-mobile");
            if (povIcon) { povIcon.classList.remove("bi-phone"); povIcon.classList.add("bi-laptop"); }
            povBtn.setAttribute("title", "Switch to Mobile view");
          }
        }

        applyPov(isMobile);

        povBtn.addEventListener("click", function () {
          isMobile = !isMobile;
          applyPov(isMobile);
          localStorage.setItem(VIEW_KEY, isMobile ? "mobile" : "auto");
        });
      }
    }

    /* ──  SCROLL-HIDE HEADERS ───────────────────── */
    function makeScrollHider(el, hideClass, threshold) {
      if (!el) return;
      var lastY = 0, ticking = false;
      function onScroll() {
        var y = window.scrollY;
        el.classList.toggle(hideClass, y > lastY && y > threshold);
        lastY = y; ticking = false;
      }
      window.addEventListener("scroll", function () {
        if (!ticking) { requestAnimationFrame(onScroll); ticking = true; }
      }, { passive: true });
    }

    makeScrollHider(document.getElementById("adminHeader"),     "hide-on-scroll", 80);
    makeScrollHider(document.getElementById("resourcesHeader"), "hide-on-scroll", 60);
    makeScrollHider(document.getElementById("historyHeader"),   "scrolled-hide",  60);

    /* ── MOBILE NAV (resources page) ─── */
    var topNav = document.getElementById("topNav");
    if (topNav) {
      var topbar = document.querySelector(".topbar-inner");
      if (topbar) {
        var hamburger = document.createElement("button");
        hamburger.id        = "mobileNavToggle";
        hamburger.className = "mobile-nav-toggle";
        hamburger.type      = "button";
        hamburger.setAttribute("aria-label", "Toggle navigation");
        hamburger.setAttribute("aria-expanded", "false");
        hamburger.innerHTML = '<i class="bi bi-list"></i>';

        var topActions = topbar.querySelector(".top-actions");
        topbar.insertBefore(hamburger, topActions || null);

        hamburger.addEventListener("click", function () {
          var open = topNav.classList.toggle("nav-open");
          hamburger.setAttribute("aria-expanded", String(open));
          hamburger.innerHTML = open ? '<i class="bi bi-x-lg"></i>' : '<i class="bi bi-list"></i>';
        });

        topNav.querySelectorAll("a").forEach(function (link) {
          link.addEventListener("click", function () {
            topNav.classList.remove("nav-open");
            hamburger.setAttribute("aria-expanded", "false");
            hamburger.innerHTML = '<i class="bi bi-list"></i>';
          });
        });

        document.addEventListener("click", function (e) {
          if (topNav.classList.contains("nav-open") && !topNav.contains(e.target) && !hamburger.contains(e.target)) {
            topNav.classList.remove("nav-open");
            hamburger.setAttribute("aria-expanded", "false");
            hamburger.innerHTML = '<i class="bi bi-list"></i>';
          }
        });
      }
    }

  });

})();