import React from 'react';

/**
 * VisuallyHidden Component
 *
 * Hides content visually but keeps it accessible to screen readers.
 *
 * Usage:
 *   <button>
 *     <Icon />
 *     <VisuallyHidden>Close menu</VisuallyHidden>
 *   </button>
 *
 *   <VisuallyHidden as="h2">Section heading for screen readers</VisuallyHidden>
 */

const visuallyHiddenStyles = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  padding: 0,
  margin: '-1px',
  overflow: 'hidden',
  clip: 'rect(0, 0, 0, 0)',
  whiteSpace: 'nowrap',
  border: 0
};

function VisuallyHidden({ as: Component = 'span', children, ...props }) {
  return (
    <Component style={visuallyHiddenStyles} {...props}>
      {children}
    </Component>
  );
}

/**
 * CSS class for visually hidden (add to tokens.css if preferred)
 *
 * .sr-only {
 *   position: absolute;
 *   width: 1px;
 *   height: 1px;
 *   padding: 0;
 *   margin: -1px;
 *   overflow: hidden;
 *   clip: rect(0, 0, 0, 0);
 *   white-space: nowrap;
 *   border: 0;
 * }
 */

export default VisuallyHidden;
