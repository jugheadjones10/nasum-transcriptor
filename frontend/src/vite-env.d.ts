/// <reference types="vite/client" />

declare module 'abcjs' {
  export function renderAbc(
    target: HTMLElement | string,
    abc: string,
    options?: {
      responsive?: string
      add_classes?: boolean
      staffwidth?: number
      wrap?: {
        minSpacing?: number
        maxSpacing?: number
        preferredMeasuresPerLine?: number
      }
    }
  ): void
}
