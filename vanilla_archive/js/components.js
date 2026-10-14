/**
 * DAD Company — Components
 * Tabs, Drawers, Dialogs
 */

class Components {
  constructor() {
    this.initTabs();
    this.initDrawers();
    this.initMobileMenu();
  }

  initTabs() {
    const tabGroups = document.querySelectorAll('.tab-group');
    
    tabGroups.forEach(group => {
      const tabs = group.querySelectorAll('.tab-group__tab');
      
      tabs.forEach(tab => {
        tab.addEventListener('click', () => {
          // Deactivate all
          tabs.forEach(t => {
            t.classList.remove('tab-group__tab--active');
            t.setAttribute('aria-selected', 'false');
            const panelId = t.getAttribute('data-panel');
            const panel = document.getElementById(panelId);
            if (panel) {
              panel.classList.remove('tab-group__panel--active');
              panel.hidden = true;
            }
          });

          // Activate clicked
          tab.classList.add('tab-group__tab--active');
          tab.setAttribute('aria-selected', 'true');
          const panelId = tab.getAttribute('data-panel');
          const panel = document.getElementById(panelId);
          if (panel) {
            panel.classList.add('tab-group__panel--active');
            panel.hidden = false;
          }
        });
      });
    });
  }

  initDrawers() {
    const triggers = document.querySelectorAll('.drawer__trigger');
    
    triggers.forEach(trigger => {
      trigger.addEventListener('click', () => {
        const drawer = trigger.closest('.drawer');
        const content = drawer.querySelector('.drawer__content');
        const inner = drawer.querySelector('.drawer__inner');
        const isOpen = drawer.classList.contains('drawer--open');
        
        // Close others (accordion style) - optional, removing for now to allow multiple open
        // document.querySelectorAll('.drawer').forEach(d => {
        //   d.classList.remove('drawer--open');
        //   d.querySelector('.drawer__content').style.height = '0px';
        //   d.querySelector('.drawer__trigger').setAttribute('aria-expanded', 'false');
        // });

        if (!isOpen) {
          drawer.classList.add('drawer--open');
          trigger.setAttribute('aria-expanded', 'true');
          content.style.height = `${inner.scrollHeight}px`;
        } else {
          drawer.classList.remove('drawer--open');
          trigger.setAttribute('aria-expanded', 'false');
          content.style.height = '0px';
        }
      });
    });
  }

  initMobileMenu() {
    const hamburger = document.querySelector('.nav__hamburger');
    const dialog = document.getElementById('mobile-menu');
    
    if (hamburger && dialog) {
      hamburger.addEventListener('click', () => {
        dialog.showModal();
      });
    }
  }
}

window.Components = Components;
