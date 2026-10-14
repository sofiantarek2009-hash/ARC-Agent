/**
 * DAD Company — Navigation
 * Scroll Spy, Smooth Scroll
 */

class Navigation {
  constructor() {
    this.initSmoothScroll();
    this.initScrollSpy();
  }

  initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', function (e) {
        const targetId = this.getAttribute('href');
        if (targetId === '#') return;
        
        const target = document.querySelector(targetId);
        if (target) {
          e.preventDefault();
          const headerOffset = 72;
          const elementPosition = target.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.scrollY - headerOffset;
  
          window.scrollTo({
             top: offsetPosition,
             behavior: "smooth"
          });
        }
      });
    });
  }

  initScrollSpy() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav__link[data-section]');

    const observerOptions = {
      root: null,
      rootMargin: '-73px 0px -70% 0px', // Adjusted for 72px header
      threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const id = entry.target.getAttribute('id');
          navLinks.forEach(link => {
            if (link.getAttribute('data-section') === id) {
              link.classList.add('nav__link--active');
            } else {
              link.classList.remove('nav__link--active');
            }
          });
        }
      });
    }, observerOptions);

    sections.forEach(section => observer.observe(section));
  }
}

window.Navigation = Navigation;
