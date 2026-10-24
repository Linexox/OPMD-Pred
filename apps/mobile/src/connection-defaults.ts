// Refresh previously shipped local endpoints once when the default deployment changes.
export function shouldRefreshConnection(savedUrl: string, packagedUrl: string): boolean {
  try {
    const saved = new URL(savedUrl)
    const packaged = new URL(packagedUrl)
    if (saved.href.replace(/\/$/, '') === packaged.href.replace(/\/$/, '')) return true
    return ['127.0.0.1', 'localhost', '10.0.2.2', '[::1]'].includes(saved.hostname)
      && saved.port === '8765' && saved.pathname === '/' && !saved.search && !saved.hash
  } catch {
    return !savedUrl.trim()
  }
}
