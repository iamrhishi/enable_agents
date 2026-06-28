import { useCallback, useEffect, useState } from 'react';

/**
 * useKeyboard Hook
 *
 * Provides keyboard navigation for lists, menus, and other interactive elements.
 *
 * Usage:
 *   const { activeIndex, handleKeyDown, setActiveIndex } = useKeyboard({
 *     itemCount: items.length,
 *     onSelect: (index) => handleItemSelect(items[index]),
 *     orientation: 'vertical'
 *   });
 */

function useKeyboard({
  itemCount,
  onSelect,
  onEscape,
  orientation = 'vertical', // 'vertical' | 'horizontal' | 'both'
  loop = true,
  initialIndex = -1
}) {
  const [activeIndex, setActiveIndex] = useState(initialIndex);

  const handleKeyDown = useCallback((e) => {
    if (itemCount === 0) return;

    const isVertical = orientation === 'vertical' || orientation === 'both';
    const isHorizontal = orientation === 'horizontal' || orientation === 'both';

    let nextIndex = activeIndex;
    let handled = false;

    switch (e.key) {
      case 'ArrowDown':
        if (isVertical) {
          e.preventDefault();
          handled = true;
          if (activeIndex < itemCount - 1) {
            nextIndex = activeIndex + 1;
          } else if (loop) {
            nextIndex = 0;
          }
        }
        break;

      case 'ArrowUp':
        if (isVertical) {
          e.preventDefault();
          handled = true;
          if (activeIndex > 0) {
            nextIndex = activeIndex - 1;
          } else if (loop) {
            nextIndex = itemCount - 1;
          }
        }
        break;

      case 'ArrowRight':
        if (isHorizontal) {
          e.preventDefault();
          handled = true;
          if (activeIndex < itemCount - 1) {
            nextIndex = activeIndex + 1;
          } else if (loop) {
            nextIndex = 0;
          }
        }
        break;

      case 'ArrowLeft':
        if (isHorizontal) {
          e.preventDefault();
          handled = true;
          if (activeIndex > 0) {
            nextIndex = activeIndex - 1;
          } else if (loop) {
            nextIndex = itemCount - 1;
          }
        }
        break;

      case 'Home':
        e.preventDefault();
        handled = true;
        nextIndex = 0;
        break;

      case 'End':
        e.preventDefault();
        handled = true;
        nextIndex = itemCount - 1;
        break;

      case 'Enter':
      case ' ':
        if (activeIndex >= 0 && onSelect) {
          e.preventDefault();
          handled = true;
          onSelect(activeIndex);
        }
        break;

      case 'Escape':
        if (onEscape) {
          e.preventDefault();
          handled = true;
          onEscape();
        }
        break;

      default:
        break;
    }

    if (handled && nextIndex !== activeIndex) {
      setActiveIndex(nextIndex);
    }

    return handled;
  }, [activeIndex, itemCount, loop, onSelect, onEscape, orientation]);

  // Reset active index when item count changes
  useEffect(() => {
    if (activeIndex >= itemCount) {
      setActiveIndex(itemCount - 1);
    }
  }, [itemCount, activeIndex]);

  return {
    activeIndex,
    setActiveIndex,
    handleKeyDown,
    // Helper to check if item is active
    isActive: (index) => index === activeIndex
  };
}

/**
 * useTypeahead Hook
 *
 * Enables type-to-select in lists.
 *
 * Usage:
 *   const { handleKeyDown } = useTypeahead({
 *     items: ['Apple', 'Banana', 'Cherry'],
 *     onMatch: (index) => setSelectedIndex(index)
 *   });
 */
function useTypeahead({
  items,
  onMatch,
  getItemText = (item) => (typeof item === 'string' ? item : item.label || ''),
  timeout = 500
}) {
  const [searchString, setSearchString] = useState('');

  // Clear search string after timeout
  useEffect(() => {
    if (!searchString) return;

    const timer = setTimeout(() => {
      setSearchString('');
    }, timeout);

    return () => clearTimeout(timer);
  }, [searchString, timeout]);

  const handleKeyDown = useCallback((e) => {
    // Only handle printable characters
    if (e.key.length !== 1 || e.ctrlKey || e.metaKey || e.altKey) {
      return false;
    }

    const newSearch = searchString + e.key.toLowerCase();
    setSearchString(newSearch);

    // Find matching item
    const matchIndex = items.findIndex((item) =>
      getItemText(item).toLowerCase().startsWith(newSearch)
    );

    if (matchIndex >= 0 && onMatch) {
      onMatch(matchIndex);
    }

    return true;
  }, [items, searchString, getItemText, onMatch]);

  return {
    searchString,
    handleKeyDown,
    clearSearch: () => setSearchString('')
  };
}

/**
 * useRovingTabIndex Hook
 *
 * Manages roving tabindex for widget navigation.
 * Only the active item has tabindex="0", others have tabindex="-1".
 *
 * Usage:
 *   const { getTabIndex, handleKeyDown, activeIndex } = useRovingTabIndex({
 *     itemCount: tabs.length
 *   });
 *
 *   {tabs.map((tab, i) => (
 *     <button tabIndex={getTabIndex(i)} onKeyDown={handleKeyDown}>
 *       {tab.label}
 *     </button>
 *   ))}
 */
function useRovingTabIndex({
  itemCount,
  orientation = 'horizontal',
  loop = true,
  initialIndex = 0
}) {
  const [activeIndex, setActiveIndex] = useState(initialIndex);

  const handleKeyDown = useCallback((e) => {
    const isVertical = orientation === 'vertical';
    const prevKey = isVertical ? 'ArrowUp' : 'ArrowLeft';
    const nextKey = isVertical ? 'ArrowDown' : 'ArrowRight';

    let nextIndex = activeIndex;

    if (e.key === nextKey) {
      e.preventDefault();
      nextIndex = loop
        ? (activeIndex + 1) % itemCount
        : Math.min(activeIndex + 1, itemCount - 1);
    } else if (e.key === prevKey) {
      e.preventDefault();
      nextIndex = loop
        ? (activeIndex - 1 + itemCount) % itemCount
        : Math.max(activeIndex - 1, 0);
    } else if (e.key === 'Home') {
      e.preventDefault();
      nextIndex = 0;
    } else if (e.key === 'End') {
      e.preventDefault();
      nextIndex = itemCount - 1;
    }

    if (nextIndex !== activeIndex) {
      setActiveIndex(nextIndex);
    }
  }, [activeIndex, itemCount, loop, orientation]);

  const getTabIndex = useCallback((index) => {
    return index === activeIndex ? 0 : -1;
  }, [activeIndex]);

  return {
    activeIndex,
    setActiveIndex,
    handleKeyDown,
    getTabIndex
  };
}

export { useKeyboard, useTypeahead, useRovingTabIndex };
export default useKeyboard;
