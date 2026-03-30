/**
 * 生成 UTF-8 的 zh-CN.json（源码仅用 \\u 转义，避免编辑器/工具链写坏中文）
 */
import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const out = path.join(__dirname, '../src/locales/zh-CN.json')

const data = {
  common: {
    lang: {
      zh: '\u4e2d\u6587',
      en: 'English',
      switchAria: '\u5207\u6362\u8bed\u8a00',
    },
    buttons: { close: '\u5173\u95ed' },
    view: {
      gridView: '\u7f51\u683c\u89c6\u56fe',
      tableView: '\u8868\u683c\u89c6\u56fe',
    },
    pagination: {
      previous: '\u4e0a\u4e00\u9875',
      next: '\u4e0b\u4e00\u9875',
      first: '\u9996\u9875',
      last: '\u5c3e\u9875',
      pagePrefix: '\u7b2c',
      pageSuffix: '/ {{total}} \u9875',
      total: '\u5171 {{total}} \u6761\u8bb0\u5f55',
      pageSize: '\u6bcf\u9875\u663e\u793a',
      items: '\u6761',
    },
  },
  plugins: {
    title: '\u63d2\u4ef6\u7ba1\u7406',
    searchPlaceholder: '\u641c\u7d22\u63d2\u4ef6\u540d\u79f0\u3001\u7b80\u4ecb',
    tabs: { market: '\u63d2\u4ef6\u5e02\u573a' },
    tableView: {
      columns: {
        plugin: '\u63d2\u4ef6',
        type: '\u8fd0\u884c\u65f6',
        publisher: '\u53d1\u5e03\u8005',
        version: '\u7248\u672c',
        viewCount: '\u6d4f\u89c8',
        installCount: '\u5b89\u88c5',
        likeCount: '\u70b9\u8d5e',
        actions: '\u64cd\u4f5c',
      },
    },
    metrics: {
      view: '\u6d4f\u89c8',
      install: '\u5b89\u88c5',
      like: '\u70b9\u8d5e',
    },
    filters: { allCategories: '\u5168\u90e8\u8fd0\u884c\u65f6' },
    viewMode: { grid: '\u7f51\u683c\u89c6\u56fe', list: '\u5217\u8868\u89c6\u56fe' },
    noDescription: '\u6682\u65e0\u63cf\u8ff0',
    noMatching: '\u672a\u627e\u5230\u5339\u914d\u7684\u63d2\u4ef6',
    noMatchingDescription: '\u5c1d\u8bd5\u8c03\u6574\u641c\u7d22\u6216\u7b5b\u9009\u6761\u4ef6',
    loading: '\u52a0\u8f7d\u4e2d...',
    actions: {
      view: '\u8be6\u60c5',
      refresh: '\u5237\u65b0',
    },
    runtime: {
      tools: 'Tools',
      mcpStdio: 'MCP Stdio',
      restfulApi: 'RESTful API',
      unknown: '\u672a\u77e5\u7c7b\u578b\uff1a{{type}}',
    },
    detail: {
      summary: '\u7b80\u4ecb',
      description: '\u8be6\u7ec6\u8bf4\u660e',
      publisher: '\u53d1\u5e03\u8005',
      runtime: '\u8fd0\u884c\u65f6',
      version: '\u6700\u65b0\u7248\u672c',
      rating: '\u8bc4\u5206',
      installCount: '\u4e0b\u8f7d/\u5b89\u88c5\u91cf',
      reviewCount: '\u8bc4\u8bba\u6570',
      viewCount: '\u6d4f\u89c8\u91cf',
      likeCount: '\u70b9\u8d5e\u6570',
      createTime: '\u521b\u5efa\u65f6\u95f4',
      updateTime: '\u66f4\u65b0\u65f6\u95f4',
      tags: '\u6807\u7b7e',
    },
  },
}

fs.writeFileSync(out, JSON.stringify(data, null, 2) + '\n', 'utf8')
console.log('Wrote', out)
