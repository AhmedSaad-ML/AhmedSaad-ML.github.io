---
# Leave the homepage title empty to use the site title
title: ""
date: 2022-10-24
type: landing

design:
  # Default section spacing
  spacing: "4rem"

sections:
  - block: resume-biography-3

    content:
      # Choose a user profile to display (a folder name within `content/authors/`)
      username: admin
      text: ""

      # Show a call-to-action button under your biography? (optional)
      button:
        text: Show CV
        url: /CV_Saad.pdf

    design:
      css_class: ""
      custom_style: |
        height: 700px;        /* Set a taller block height */
        width: 100%;          /* Ensure full-width */
        background-size: cover;  /* Make the background image cover the whole area */
        background-position: center; /* Center the background */
        background-repeat: no-repeat; /* Avoid repeating the image */

    background:
      color: black
      image:
        filename: your-image-file.jpg  # Replace with your actual image filename
      filters:
        brightness: 1.0
      size: cover  # Force the image to cover the block area
      position: center
      parallax: false
  - block: collection
    content:
      title: Publications
      text: ""
      filters:
        folders:
          - publication
        exclude_featured: false
    design:
      view: citation
    ---
