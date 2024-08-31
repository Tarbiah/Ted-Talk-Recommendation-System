// script.js
var prevScrollPos = window.pageYOffset;

window.onscroll = function() {
  var currentScrollPos = window.pageYOffset;

  if (prevScrollPos > currentScrollPos) {
    document.querySelector('header').style.top = '0';
  } else {
    document.querySelector('header').style.top = '-70px'; /* Adjust the height of your header */
  }

  prevScrollPos = currentScrollPos;
};

window.onscroll = function() {
    const header = document.getElementById('your-header-id'); // Replace with your actual header ID
    const sticky = header.offsetTop;

    if (window.pageYOffset > sticky) {
        header.classList.add('fixed-header');
    } else {
        header.classList.remove('fixed-header');
    }
};
