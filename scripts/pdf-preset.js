/*
 * Print preset used to render the lecture decks to PDF.
 *
 * scripts/pdf.mjs injects this file into a deck before any of the deck's own
 * scripts run, and navigates with reveal's ?print-pdf flag plus a few config
 * overrides in the query string. Nothing here changes how a deck looks on
 * screen; it prepares reveal's print layout for one headless printToPDF call.
 *
 * The part reveal cannot do for itself is the ordering. Its print module lays
 * every page out (heights, vertical centring, clipping) as soon as the window
 * loads, while MathJax is still fetching its TeX fonts and typesetting, so a
 * slide whose equations turn out taller than the space measured for it ends up
 * off-centre or clipped. settle() waits for the whole chain, and paginate()
 * then re-measures the pages with the finished layout.
 */
(function () {
  'use strict';

  // The deck footer sits outside .slides, so reveal leaves it hanging below
  // the last page. paginate() clones it into each page instead. The last page
  // keeps reveal's page-break-after rule off, which otherwise prints a blank
  // page after the deck.
  var PRINT_CSS = [
    'html.print-pdf .reveal > .deck-footer { display: none !important; }',
    'html.print-pdf .reveal .slides > .pdf-page:last-child {',
    '  page-break-after: avoid !important; break-after: avoid !important;',
    '}'
  ].join('\n');

  var DEFAULTS = {
    // Re-typeset the whole deck once the print layout and the fonts are in
    // place, so equations are measured against their final surroundings.
    reprocessMath: true,
    // Repeat each deck's .deck-footer line on every page.
    footer: true,
    // Per-step ceiling for the waits below.
    timeoutMs: 120000
  };

  var pdfReadyFired = false;
  var onPdfReady;
  var pdfReady = new Promise(function (resolve) { onPdfReady = resolve; });
  document.addEventListener('pdf-ready', function () {
    pdfReadyFired = true;
    onPdfReady();
  }, true);

  function sleep(ms) {
    return new Promise(function (resolve) { setTimeout(resolve, ms); });
  }

  function withTimeout(promise, ms, label) {
    return Promise.race([
      promise,
      sleep(ms).then(function () {
        throw new Error('timed out waiting for ' + label);
      })
    ]);
  }

  function poll(until, ms, label) {
    return withTimeout(new Promise(function (resolve) {
      (function step() {
        var value = until();
        if (value) { resolve(value); return; }
        setTimeout(step, 50);
      })();
    }), ms, label);
  }

  function slidesElement() {
    var slides = document.querySelector('.reveal .slides');
    if (!slides) throw new Error('this page has no .slides container');
    return slides;
  }

  function config() {
    if (!window.Reveal || !Reveal.getConfig) {
      throw new Error('reveal.js never initialised');
    }
    return Reveal.getConfig();
  }

  /*
   * The @page size reveal's print module injects. Reading it back keeps this
   * script in step with reveal's own page-size formula; the fallback is that
   * formula (canvas plus the configured margin).
   */
  function pageSize(config) {
    var fallback = {
      width: Math.floor(config.width * (1 + config.margin)),
      height: Math.floor(config.height * (1 + config.margin))
    };
    var sheets;
    try { sheets = document.styleSheets; } catch (e) { return fallback; }
    for (var i = 0; i < sheets.length; i++) {
      var rules;
      try { rules = sheets[i].cssRules; } catch (e) { continue; }
      if (!rules) continue;
      for (var j = 0; j < rules.length; j++) {
        var rule = rules[j];
        if (!(rule instanceof CSSPageRule)) continue;
        var match = /size:\s*([\d.]+)px\s+([\d.]+)px/.exec(rule.cssText);
        if (match) {
          return { width: parseFloat(match[1]), height: parseFloat(match[2]) };
        }
      }
    }
    return fallback;
  }

  function applyPrintCss() {
    if (document.getElementById('phys-pdf-preset')) return;
    var style = document.createElement('style');
    style.id = 'phys-pdf-preset';
    style.textContent = PRINT_CSS;
    document.head.appendChild(style);
  }

  function whenFontsLoaded(timeoutMs) {
    if (!document.fonts) return Promise.resolve({ status: 'unsupported', faces: 0 });
    return withTimeout(document.fonts.ready, timeoutMs, 'webfonts').then(function () {
      return { status: document.fonts.status, faces: document.fonts.size };
    });
  }

  function whenImagesLoaded(slides, timeoutMs) {
    var images = Array.prototype.slice.call(slides.querySelectorAll('img'));
    var pending = images.filter(function (img) {
      return !(img.complete && img.naturalWidth);
    });
    return withTimeout(Promise.all(pending.map(function (img) {
      return new Promise(function (resolve) {
        img.addEventListener('load', resolve, { once: true });
        img.addEventListener('error', resolve, { once: true });
      });
    })), timeoutMs, 'the deck images').then(function () {
      return { total: images.length, unloaded: pending.length };
    });
  }

  function mathHub() {
    return (window.MathJax && window.MathJax.Hub) || null;
  }

  // tex2jax keeps the source in a hidden preview span, so this stays true
  // after typesetting; it only answers "does this deck contain TeX at all".
  function deckUsesMath(slides) {
    return /\$\$|\\\(|\\\[/.test(slides.textContent || '');
  }

  function drainMathJaxQueue(hub, timeoutMs) {
    return withTimeout(new Promise(function (resolve) {
      hub.Queue(resolve);
    }), timeoutMs, 'the MathJax queue');
  }

  /*
   * Re-typeset from the stored TeX. MathJax's first pass can measure against
   * fallback fonts or a layout that was about to change; a second pass after
   * fonts.ready is what makes the printed equations match the screen ones.
   */
  function reprocessMath(hub, slides) {
    if (typeof hub.Reprocess === 'function') {
      hub.Queue(['Reprocess', hub, slides]);
    } else {
      hub.Queue(['Typeset', hub, slides]);
    }
  }

  function headingOf(slide) {
    var heading = slide.querySelector('h1, h2, h3');
    var text = heading ? heading.textContent.trim() : '';
    return text.length > 60 ? text.slice(0, 57) + '...' : text;
  }

  /*
   * Waits until the deck is as finished as it is going to get, and reports the
   * page size to print at.
   */
  async function settle(options) {
    var opts = Object.assign({}, DEFAULTS, options);
    var slides = slidesElement();

    await poll(function () {
      return window.Reveal && Reveal.isReady && Reveal.isReady();
    }, opts.timeoutMs, 'reveal to become ready');
    if (!pdfReadyFired) {
      await withTimeout(pdfReady, opts.timeoutMs, "reveal's pdf-ready event");
    }

    var deckConfig = config();
    var size = pageSize(deckConfig);
    applyPrintCss();

    var images = await whenImagesLoaded(slides, opts.timeoutMs);
    var fonts = await whenFontsLoaded(opts.timeoutMs);

    var math = 'none';
    if (mathHub() || deckUsesMath(slides)) {
      var hub = await poll(mathHub, opts.timeoutMs, 'MathJax to load');
      await drainMathJaxQueue(hub, opts.timeoutMs);
      if (opts.reprocessMath) {
        reprocessMath(hub, slides);
        await drainMathJaxQueue(hub, opts.timeoutMs);
        // MathJax asks for its TeX fonts while typesetting, so give the
        // browser another chance to fetch them before measuring.
        fonts = await whenFontsLoaded(opts.timeoutMs);
        math = 'reprocessed';
      } else {
        math = 'typeset';
      }
    }

    return {
      title: document.title,
      url: location.href,
      pageWidthPx: size.width,
      pageHeightPx: size.height,
      slideWidthPx: deckConfig.width,
      slideHeightPx: deckConfig.height,
      math: math,
      fonts: fonts,
      images: images
    };
  }

  /*
   * One PDF page per slide: re-measure each page now that the equations have
   * their final size, shrink anything that no longer fits, re-centre it, and
   * stamp the footer.
   */
  function paginate(options) {
    var opts = Object.assign({}, DEFAULTS, options);
    var deckConfig = config();
    var size = pageSize(deckConfig);
    var footer = document.querySelector('.reveal > .deck-footer');
    var pages = Array.prototype.slice.call(
      document.querySelectorAll('.reveal .slides > .pdf-page')
    );
    var shrunk = [];

    pages.forEach(function (page, index) {
      var slide = page.querySelector(':scope > section');
      if (!slide || slide.classList.contains('stack')) return;

      if (opts.footer && footer && !page.querySelector(':scope > .deck-footer')) {
        var stamp = footer.cloneNode(true);
        stamp.classList.add('pdf-page-footer');
        stamp.removeAttribute('id');
        page.appendChild(stamp);
      }

      var content = Math.ceil(slide.scrollHeight);
      var width = Math.ceil(slide.scrollWidth);
      // A slide taller than its page box prints with everything below the
      // heading missing, so shrink those to fit the canvas. zoom reflows the
      // layout; a transform would leave the box, and so the print clip, in
      // the same oversized place.
      var scale = content > 0 && width > 0
        ? Math.min(1, deckConfig.width / width, deckConfig.height / content)
        : 1;
      if (scale < 1) slide.style.zoom = String(scale);

      var shown = Math.ceil(content * scale);
      page.style.height = size.height + 'px';
      slide.style.top =
        Math.max(Math.floor((size.height - shown) / 2), 0) + 'px';

      if (scale < 1) {
        shrunk.push({
          page: index + 1,
          heading: headingOf(slide),
          contentWidth: width,
          contentHeight: content,
          canvasWidth: deckConfig.width,
          canvasHeight: deckConfig.height,
          percent: Math.round(scale * 100)
        });
      }
    });

    return {
      pages: pages.length,
      pageWidthPx: size.width,
      pageHeightPx: size.height,
      shrunk: shrunk
    };
  }

  window.PhysPdf = {
    settle: settle,
    paginate: paginate
  };
})();
