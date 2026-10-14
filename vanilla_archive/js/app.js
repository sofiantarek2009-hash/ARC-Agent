/**
 * DAD Company — Main Application Entry
 * Initializes all modules on DOMContentLoaded
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Modules
  if (window.Animations) new window.Animations();
  if (window.Components) new window.Components();
  if (window.Navigation) new window.Navigation();

  console.log('DAD Company — Single Brand Architecture initialized.');
});
