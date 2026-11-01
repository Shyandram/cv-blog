/** Public chapter paths stay stable when the course tree changes its numbering. */
export function chapterSlug(slug) {
  return slug.replace(/^([^/]+)\/(?:dip|dl|ml|pr|nlp|opt|it)-\d+[a-z]?-/i, '$1/');
}
export function chapterPath(path) {
  return path.replace(/(\/(?:en|zh-tw)\/learn\/)([^/#]+\/[^/#]+)/g, (_, base, slug) => base + chapterSlug(slug));
}
