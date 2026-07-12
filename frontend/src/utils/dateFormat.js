/**
 * Date Format Utility
 *
 * Centralized date formatting for the entire app.
 * Format is stored in localStorage and can be changed in settings.
 */

const DATE_FORMAT_KEY = 'enableAgentsDateFormat';

// Available formats
export const DATE_FORMATS = {
  'DD/MM/YYYY': 'dd/mm/yyyy',
  'MM/DD/YYYY': 'mm/dd/yyyy',
  'YYYY-MM-DD': 'yyyy-mm-dd',
};

/**
 * Get the current date format preference
 */
export const getDateFormat = () => {
  return localStorage.getItem(DATE_FORMAT_KEY) || 'DD/MM/YYYY';
};

/**
 * Set the date format preference
 */
export const setDateFormat = (format) => {
  localStorage.setItem(DATE_FORMAT_KEY, format);
  // Dispatch event for components to update
  window.dispatchEvent(new Event('dateFormatChange'));
};

/**
 * Format a date string or Date object according to user preference
 * @param {string|Date} date - The date to format
 * @returns {string} Formatted date string
 */
export const formatDate = (date) => {
  if (!date) return '';

  const d = typeof date === 'string' ? new Date(date) : date;

  if (isNaN(d.getTime())) return date; // Return original if invalid

  const day = String(d.getDate()).padStart(2, '0');
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const year = d.getFullYear();

  const format = getDateFormat();

  switch (format) {
    case 'MM/DD/YYYY':
      return `${month}/${day}/${year}`;
    case 'YYYY-MM-DD':
      return `${year}-${month}-${day}`;
    case 'DD/MM/YYYY':
    default:
      return `${day}/${month}/${year}`;
  }
};

/**
 * Format a date with time
 * @param {string|Date} date - The date to format
 * @returns {string} Formatted date and time string
 */
export const formatDateTime = (date) => {
  if (!date) return '';

  const d = typeof date === 'string' ? new Date(date) : date;

  if (isNaN(d.getTime())) return date;

  const dateStr = formatDate(d);
  const hours = String(d.getHours()).padStart(2, '0');
  const minutes = String(d.getMinutes()).padStart(2, '0');

  return `${dateStr} ${hours}:${minutes}`;
};

/**
 * Format time only (no seconds) - HH:MM AM/PM
 * @param {string|Date} date - The date to format
 * @returns {string} Formatted time string
 */
export const formatTime = (date) => {
  if (!date) return '';

  const d = typeof date === 'string' ? new Date(date) : date;

  if (isNaN(d.getTime())) return '';

  let hours = d.getHours();
  const minutes = String(d.getMinutes()).padStart(2, '0');
  const ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12;
  hours = hours ? hours : 12; // 0 should be 12

  return `${hours}:${minutes} ${ampm}`;
};

/**
 * Get relative date label (Today, Yesterday, or formatted date)
 * @param {string|Date} date - The date to check
 * @returns {string} Relative date label
 */
export const getRelativeDateLabel = (date) => {
  if (!date) return '';

  const d = typeof date === 'string' ? new Date(date) : date;

  if (isNaN(d.getTime())) return '';

  const today = new Date();
  const yesterday = new Date(today);
  yesterday.setDate(yesterday.getDate() - 1);

  const isToday = d.toDateString() === today.toDateString();
  const isYesterday = d.toDateString() === yesterday.toDateString();

  if (isToday) return 'Today';
  if (isYesterday) return 'Yesterday';
  return formatDate(d);
};

/**
 * Check if two dates are on the same day
 * @param {string|Date} date1
 * @param {string|Date} date2
 * @returns {boolean}
 */
export const isSameDay = (date1, date2) => {
  if (!date1 || !date2) return false;

  const d1 = typeof date1 === 'string' ? new Date(date1) : date1;
  const d2 = typeof date2 === 'string' ? new Date(date2) : date2;

  return d1.toDateString() === d2.toDateString();
};
