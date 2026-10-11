<script setup lang="ts">
defineProps<{ name: string }>()
const paths: Record<string, string> = {
 home: 'M3 10 12 3l9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1Z',
 records: 'M7 3h10a2 2 0 0 1 2 2v16H5V5a2 2 0 0 1 2-2ZM9 8h6M9 12h6M9 16h4',
 user: 'M16 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0ZM4 21v-2a8 8 0 0 1 16 0v2',
 plus: 'M12 5v14M5 12h14', back: 'm14 5-7 7 7 7', arrow: 'M5 12h14m-6-6 6 6-6 6',
 camera: 'M3 7h4l2-3h6l2 3h4v14H3ZM16 14a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z',
 check: 'm5 12 4 4L19 6', calendar: 'M4 6h16v15H4ZM7 3v6m10-6v6M4 11h16',
 file: 'M5 3h9l5 5v13H5ZM14 3v5h5M9 13h6m-6 4h6', bell: 'M6 8a6 6 0 0 1 12 0v6l2 3H4l2-3Zm4 13h4',
 link: 'm9 15 6-6m-4-3 3-3a4 4 0 0 1 6 6l-3 3m-4 6-3 3a4 4 0 0 1-6-6l3-3',
 chevron: 'm9 5 7 7-7 7', help: 'M9 9a3 3 0 1 1 5 2c-2 1-2 2-2 3m0 3h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z',
 stethoscope: 'M5 3v6a5 5 0 0 0 10 0V3m-12 0h4m6 0h4M10 14v3a4 4 0 0 0 8 0v-2m3-3a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z',
 clock: 'M12 6v6l4 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z',
}
</script>
<template><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path :d="paths[name] || paths.file" /></svg></template>
