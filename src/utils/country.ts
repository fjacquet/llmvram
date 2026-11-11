const names = new Intl.DisplayNames(['en'], { type: 'region' })

export function countryName(code: string): string {
  return names.of(code) ?? code
}
