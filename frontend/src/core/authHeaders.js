/** Merge JSON headers with optional Bearer session token from login/register/OAuth */

export function authJsonHeaders(extra = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...extra,
  };
  const token = typeof localStorage !== 'undefined'
    ? localStorage.getItem('sessionToken')
    : null;
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

/** For GET requests: only add Authorization when logged in. */
export function authOptionalHeaders(extra = {}) {
  const headers = { ...extra };
  const token = typeof localStorage !== 'undefined'
    ? localStorage.getItem('sessionToken')
    : null;
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}
