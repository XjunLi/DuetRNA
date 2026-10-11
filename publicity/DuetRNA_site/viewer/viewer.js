/* Preserve the original, embedded PDB data; never execute the legacy HTML. */
(() => {
  'use strict';
  const samples = {
    '40': ['40nt_genonly.html', '0.586'], '70': ['70nt_genonly.html', '0.712'],
    '90': ['90nt_genonly.html', '0.765'], '110': ['110nt_genonly.html', '0.863'],
    '120': ['120nt_s16_genonly.html', '0.974'], '140': ['140nt_genonly.html', '0.679']
  };
  const length = new URLSearchParams(location.search).get('length') || '120';
  const status = document.getElementById('viewer-status');
  const retry = document.getElementById('retry-view');
  const reset = document.getElementById('reset-view');
  const download = document.getElementById('download-pdb');
  let viewer = null, controller = null, pdbURL = '', disposed = false;
  const notify = type => { if (parent !== window) parent.postMessage({ type }, location.origin); };
  if (!samples[length]) {
    status.textContent = 'Unknown sample. Available lengths: 40, 70, 90, 110, 120 and 140 nt.';
    notify('duetrna:error'); return;
  }
  document.title = `DuetRNA · ${length} nt`;
  document.getElementById('structure-title').textContent = `${length} nt · scTM ${samples[length][1]}`;

  const bases = ['https://cdn.jsdelivr.net/npm/molstar@3.2.0/build/viewer/', 'https://unpkg.com/molstar@3.2.0/build/viewer/'];
  function loadAsset(url, kind) {
    return new Promise((resolve, reject) => {
      const el = document.createElement(kind === 'css' ? 'link' : 'script');
      let done = false;
      const finish = (error) => {
        if (done) return; done = true; clearTimeout(timeout); el.onload = el.onerror = null;
        if (error) { el.remove(); reject(error); } else resolve(el);
      };
      const timeout = setTimeout(() => finish(new Error('Renderer download timed out.')), 15000);
      el.onload = () => finish(); el.onerror = () => finish(new Error('Renderer download failed.'));
      if (kind === 'css') { el.rel = 'stylesheet'; el.href = url; }
      else { el.src = url; el.async = true; }
      document.head.append(el);
    });
  }
  async function loadRenderer() {
    let error;
    for (const base of bases) {
      try {
        await loadAsset(base + 'molstar.css', 'css');
        if (!window.molstar?.Viewer) await loadAsset(base + 'molstar.js', 'js');
        if (!window.molstar?.Viewer) throw new Error('Renderer API is unavailable.');
        return;
      } catch (e) { error = e; }
    }
    throw error;
  }
  async function loadPDB() {
    controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(samples[length][0], { signal: controller.signal });
      if (!response.ok) throw new Error(`Structure data returned HTTP ${response.status}.`);
      const html = await response.text();
      const match = html.match(/atob\(\s*["']([A-Za-z0-9+/=\s]+)["']\s*\)/);
      if (!match) throw new Error('The embedded PDB structure was not found.');
      const pdb = atob(match[1].replace(/\s/g, ''));
      if (!/^ATOM\s/m.test(pdb)) throw new Error('The structure contains no ATOM records.');
      if (disposed) throw new Error('Viewer closed.');
      pdbURL = URL.createObjectURL(new Blob([pdb], { type: 'chemical/x-pdb' }));
      download.href = pdbURL; download.download = `DuetRNA_${length}nt.pdb`; download.hidden = false;
      return pdb;
    } finally { clearTimeout(timeout); }
  }
  function setActivity() {
    if (!viewer) return;
    if (document.hidden) viewer.plugin.animationLoop?.stop();
    else viewer.plugin.animationLoop?.start();
  }
  async function start() {
    try {
      const [pdb] = await Promise.all([loadPDB(), loadRenderer()]);
      if (disposed) return;
      const created = await molstar.Viewer.create('viewport', {
        extensions: [], layoutIsExpanded: false, layoutShowControls: false,
        layoutShowRemoteState: false, layoutShowSequence: false, layoutShowLog: false,
        layoutShowLeftPanel: false, volumeStreamingDisabled: true, pixelScale: 1,
        viewportShowExpand: true, viewportShowControls: true, viewportShowSettings: true
      });
      if (disposed) { created.plugin.dispose(); return; }
      viewer = created;
      viewer.plugin.canvas3d?.setProps({ renderer: { backgroundColor: 0xf8f7f2 } });
      await viewer.loadStructureFromData(pdb, 'pdb', { dataLabel: `DuetRNA ${length} nt` });
      if (disposed) return;
      status.textContent = 'Generated structure loaded.'; status.hidden = true;
      reset.disabled = false; setActivity(); notify('duetrna:ready');
    } catch (error) {
      if (disposed) return;
      status.hidden = false;
      status.textContent = `Could not open the 3D view. ${error.message || 'Please retry.'} You can download the PDB once the structure data are available.`;
      retry.hidden = false; notify('duetrna:error');
    }
  }
  retry.addEventListener('click', () => location.reload());
  reset.addEventListener('click', () => viewer?.plugin.managers.camera.reset());
  document.addEventListener('visibilitychange', setActivity);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') notify('duetrna:escape'); });
  window.addEventListener('pagehide', () => {
    disposed = true; controller?.abort(); viewer?.plugin.dispose();
    if (pdbURL) URL.revokeObjectURL(pdbURL);
  });
  // A standalone viewer restored from the back-forward cache needs a fresh context.
  window.addEventListener('pageshow', event => { if (event.persisted && disposed) location.reload(); });
  start();
})();
