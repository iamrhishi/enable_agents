/** Merge JSON headers with optional Bearer session token from login/register/OAuth */

export function authJsonHeaders(extra = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...extra,
  };

  if (typeof localStorage !== 'undefined') {
    // Add session token for auth
    const token = localStorage.getItem('sessionToken');
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    // Add X-User-Id for routes that use it directly
    const userEmail = localStorage.getItem('userEmail');
    if (userEmail) {
      headers['X-User-Id'] = userEmail;
    }
  }

  return headers;
}

/** For GET requests: only add Authorization when logged in. */
export function authOptionalHeaders(extra = {}) {
  const headers = { ...extra };

  if (typeof localStorage !== 'undefined') {
    const token = localStorage.getItem('sessionToken');
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    const userEmail = localStorage.getItem('userEmail');
    if (userEmail) {
      headers['X-User-Id'] = userEmail;
    }
  }

  return headers;
}
