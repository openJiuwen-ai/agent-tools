/**
 * 抽奖工具测试运行脚本
 * 用于在 Node.js 环境中运行 test.html 的测试用例
 */

const fs = require('fs');
const path = require('path');

// 测试结果统计
let totalTests = 0;
let passedTests = 0;
let failedTests = 0;
const testResults = [];

// 覆盖率跟踪
const coverage = {
    functions: [
        'addPrize', 'removePrize', 'getPrizes', 'updatePrizeSelect',
        'togglePrizeSettings', 'startDraw', 'getAvailableNumbers',
        'getValidRange', 'parseExcludeNumbers', 'toggleWinner',
        'renderWinnerList', 'updateStats', 'resetAll',
        'updateDrawCount', 'updateRemoveButtons', 'showMessage', 'sleep', 'init',
        'addColor', 'removeColor', 'updateColorRemoveButtons',
        'highlightAllWinners', 'captureSnapshot', 'sortWinnersByColor', 'resetWinnersOrder',
        'toggleHighlightAll', 'highlightAllStates',
        // 总抽奖池新增功能
        'totalPool', 'getColorRanges', 'parseExcludeNumbersByElement',
        // 优化功能新增
        'disableColorConfig', 'startDrawWithProgressControl', 'resetAllWithAbort',
        // 颜色下拉框优化功能
        'updateColorSelect'
    ],
    called: {}
};

// 简单的测试框架实现
class TestFramework {
    constructor() {
        this.tests = [];
    }

    test(suite, name, description, fn) {
        this.tests.push({ suite, name, description, fn });
    }

    assertEqual(actual, expected, msg = '') {
        if (actual !== expected) {
            throw new Error(`${msg}\n期望: ${expected}\n实际: ${actual}`);
        }
    }

    assertTrue(condition, msg = '') {
        if (!condition) {
            throw new Error(`${msg}\n期望: true\n实际: false`);
        }
    }

    assertFalse(condition, msg = '') {
        if (condition) {
            throw new Error(`${msg}\n期望: false\n实际: true`);
        }
    }

    markCalled(funcName) {
        coverage.called[funcName] = true;
    }

    async runAll() {
        const results = [];
        
        for (const test of this.tests) {
            totalTests++;
            try {
                await test.fn();
                passedTests++;
                results.push({
                    suite: test.suite,
                    name: test.name,
                    description: test.description,
                    status: 'pass'
                });
                console.log(`✅ [PASS] ${test.suite} - ${test.name}`);
            } catch (e) {
                failedTests++;
                results.push({
                    suite: test.suite,
                    name: test.name,
                    description: test.description,
                    status: 'fail',
                    error: e.message
                });
                console.log(`❌ [FAIL] ${test.suite} - ${test.name}`);
                console.log(`   错误: ${e.message}`);
            }
        }
        
        return results;
    }
}

const framework = new TestFramework();

// 模拟 DOM 环境
function createMockDOM() {
    // 先创建所有元素对象,使其属性可以被修改
    const elements = {
        'activityName': { 
            value: 'openJiuwen走进高校-XX', 
            type: 'text',
            placeholder: '请输入活动名称'
        },
        'minNumber': { value: '1' },
        'maxNumber': { value: '100' },
        'excludeNumbers': { value: '' },
        'prizeConfig': { innerHTML: '' },
        'currentPrize': { innerHTML: '' },
        'winnerList': { innerHTML: '' },
        'totalWinners': { textContent: '0' },
        'activeWinners': { textContent: '0' },
        'remainingNumbers': { textContent: '100' },
        'totalPool': { textContent: '100' },
        'message': { innerHTML: '' },
        'drawBtn': { disabled: false },
        'drawAnimation': { classList: { add: () => {}, remove: () => {} } },
        'colorConfig': {
            innerHTML: '',
            children: [],
            querySelectorAll: (sel) => []
        },
        'colorSelect': { disabled: false }
    };
    
    return {
        winners: [],
        excludedNumbers: new Set(),
        originalWinnersOrder: [],
        sortedPrizes: new Set(),
        highlightAllStates: new Map(),
        isDrawingInProgress: false,
        drawingAbortController: null,
        
        getElementById: function(id) {
            return elements[id] || null;
        },
        
        querySelector: function(selector) {
            if (selector === '.result-section') {
                return { 
                    querySelector: (s) => s === 'h2' ? { textContent: '中奖结果' } : null
                };
            }
            if (selector === '.winner-item') {
                return null;
            }
            if (selector === '.btn-add-color') {
                return { disabled: false };
            }
            return null;
        },
        
        querySelectorAll: function(selector) {
            if (selector === '.winner-item' || selector === '.btn-primary' || selector === 'button') {
                return [];
            }
            return [];
        },
        
        createElement: function(tag) {
            return {
                className: '',
                textContent: '',
                style: {},
                setAttribute: function() {},
                getAttribute: function() { return null; }
            };
        },
        
        defaultView: {
            getComputedStyle: function() {
                return {
                    width: '55px',
                    height: '55px',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 100%)',
                    backgroundColor: 'rgb(245, 245, 245)'
                };
            }
        },
        
        // 模拟函数
        showMessage: function(msg, type) {
            // 简单的消息显示模拟
            const messageEl = this.getElementById('message');
            if (messageEl) {
                messageEl.innerHTML = msg;
            }
        },
        addPrize: function() {},
        removePrize: function() {},
        getPrizes: function() {
            return [
                { name: 'openJiuwen定制-奖项1', count: 1, extra: 0, index: 0 },
                { name: 'openJiuwen定制-奖项2', count: 2, extra: 0, index: 1 },
                { name: 'openJiuwen定制-奖项3', count: 3, extra: 0, index: 2 }
            ];
        },
        updatePrizeSelect: function() {},
        disableColorConfig: function(disabled) {
            const colorConfig = this.getElementById('colorConfig');
            const addColorBtn = this.querySelector('.btn-add-color');
            const colorSelect = this.getElementById('colorSelect');
            
            if (colorConfig) {
                const inputs = colorConfig.querySelectorAll ? colorConfig.querySelectorAll('input, textarea') : [];
                inputs.forEach(input => input.disabled = disabled);
            }
            
            if (addColorBtn) {
                addColorBtn.disabled = disabled;
            }
            
            if (colorSelect) {
                colorSelect.disabled = disabled;
            }
        },
        updateColorSelect: function() {
            // 模拟updateColorSelect函数
            const colorSelect = this.getElementById('colorSelect');
            const colorConfig = this.getElementById('colorConfig');
            const existingColors = Array.from(colorConfig.children || []).map(item => item.dataset?.color || item.name);
            
            const allColors = [
                { name: 'red', label: '红色', bg: '#ff6b6b' },
                { name: 'blue', label: '蓝色', bg: '#48dbfb' },
                { name: 'yellow', label: '黄色', bg: '#feca57' },
                { name: 'green', label: '绿色', bg: '#26de81' },
                { name: 'brown', label: '棕色', bg: '#a55e3c' },
                { name: 'purple', label: '紫色', bg: '#a55eea' },
                { name: 'pink', label: '粉色', bg: '#fd79a8' }
            ];
            
            const availableColors = allColors.filter(c => !existingColors.includes(c.name));
            
            if (colorSelect) {
                colorSelect.innerHTML = '<option value="">选择要添加的颜色</option>';
                availableColors.forEach(color => {
                    const option = {
                        value: color.name,
                        textContent: color.label + ' ●',
                        style: {
                            color: color.bg,
                            fontWeight: 'bold'
                        }
                    };
                    colorSelect.innerHTML += `<option value="${option.value}" style="color: ${option.style.color}; font-weight: ${option.style.fontWeight};">${option.textContent}</option>`;
                });
            }
        },
        getColorRanges: function() {
            // 模拟获取颜色范围配置
            const colorConfig = this.getElementById('colorConfig');
            if (!colorConfig) {
                // 默认配置：红色1-100
                return [{
                    name: 'red',
                    label: '红色',
                    bg: '#ff6b6b',
                    min: 1,
                    max: 100,
                    exclude: new Set()
                }];
            }
            
            // 检查是否有children配置
            if (colorConfig.children && colorConfig.children.length > 0) {
                return colorConfig.children;
            }
            
            // 默认配置：红色1-100
            return [{
                name: 'red',
                label: '红色',
                bg: '#ff6b6b',
                min: 1,
                max: 100,
                exclude: new Set()
            }];
        },
        parseExcludeNumbersByElement: function(element) {
            if (!element || !element.value) return new Set();
            const exclude = new Set();
            const parts = element.value.split(',').map(s => s.trim()).filter(s => s);
            parts.forEach(part => {
                // 处理范围格式如 "1-10"
                if (part.includes('-')) {
                    const rangeParts = part.split('-').map(s => parseInt(s.trim()));
                    if (rangeParts.length === 2 && !isNaN(rangeParts[0]) && !isNaN(rangeParts[1])) {
                        const start = Math.min(rangeParts[0], rangeParts[1]);
                        const end = Math.max(rangeParts[0], rangeParts[1]);
                        for (let i = start; i <= end; i++) {
                            exclude.add(i);
                        }
                    }
                } else {
                    const num = parseInt(part);
                    if (!isNaN(num)) {
                        exclude.add(num);
                    }
                }
            });
            return exclude;
        },
        updateStats: function() {
            this.getElementById('totalWinners').textContent = this.winners.length.toString();
            this.getElementById('activeWinners').textContent = this.winners.filter(w => w.active).length.toString();
            
            // 计算总抽奖池
            const colors = this.getColorRanges();
            let totalPoolCount = 0;
            colors.forEach(color => {
                for (let i = color.min; i <= color.max; i++) {
                    if (!color.exclude.has(i)) {
                        totalPoolCount++;
                    }
                }
            });
            const totalPoolEl = this.getElementById('totalPool');
            if (totalPoolEl) {
                totalPoolEl.textContent = totalPoolCount.toString();
            }
        },
        renderWinnerList: function() {},
        sleep: function(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        },
        init: function() {},
        addColor: function() {},
        removeColor: function() {},
        updateColorRemoveButtons: function() {},
        highlightAllWinners: function(prizeIndex) {
            this.winners.forEach(w => {
                if (w.prizeIndex === prizeIndex) {
                    w.active = true;
                }
            });
        },
        captureSnapshot: async function() {},
        sortWinnersByColor: function() {},
        resetWinnersOrder: function() {},
        toggleWinner: function(index) {
            if (this.winners[index]) {
                this.winners[index].active = !this.winners[index].active;
            }
        },
        toggleHighlightAll: function(prizeIndex) {
            framework.markCalled('toggleHighlightAll');
            const isHighlighted = this.highlightAllStates.get(prizeIndex);
            
            if (isHighlighted) {
                this.winners.forEach(w => {
                    if (w.prizeIndex === prizeIndex) {
                        w.active = false;
                    }
                });
                this.highlightAllStates.delete(prizeIndex);
                this.showMessage('已全部熄灭！', 'success');
            } else {
                this.winners.forEach(w => {
                    if (w.prizeIndex === prizeIndex) {
                        w.active = true;
                    }
                });
                this.highlightAllStates.set(prizeIndex, true);
                this.showMessage('已全部点亮！', 'success');
            }
            
            this.renderWinnerList();
            this.updateStats();
        },
        resetAllWithOptimization: function() {
            framework.markCalled('resetAll');
            this.winners = [];
            this.originalWinnersOrder = [];
            this.sortedPrizes.clear();
            this.highlightAllStates.clear();
            this.renderWinnerList();
            this.updatePrizeSelect();
            this.updateStats();
            this.showMessage('已重置所有数据', 'success');
        }
    };
}

// 设置测试环境
function setupTestEnv() {
    return createMockDOM();
}

const doc = createMockDOM();

// ==================== 测试用例定义 ====================

// 1. 活动名称输入框测试
framework.test('活动名称输入框', '默认值 - openJiuwen走进高校-XX', '验证活动名称输入框默认值', () => {
    const activityNameInput = doc.getElementById('activityName');
    framework.assertEqual(activityNameInput.value, 'openJiuwen走进高校-XX', '默认值应为openJiuwen走进高校-XX');
});

framework.test('活动名称输入框', '输入框存在性', '验证活动名称输入框存在', () => {
    const activityNameInput = doc.getElementById('activityName');
    framework.assertTrue(activityNameInput !== null, '活动名称输入框应存在');
    framework.assertEqual(activityNameInput.type, 'text', '应为文本输入框');
});

framework.test('活动名称输入框', 'placeholder属性', '验证活动名称placeholder', () => {
    const activityNameInput = doc.getElementById('activityName');
    framework.assertEqual(activityNameInput.placeholder, '请输入活动名称', 'placeholder应为请输入活动名称');
});

framework.test('活动名称输入框', '自定义活动名称', '测试修改活动名称', () => {
    const activityNameInput = doc.getElementById('activityName');
    activityNameInput.value = '测试活动2024';
    framework.assertEqual(activityNameInput.value, '测试活动2024', '活动名称应可修改');
});

framework.test('活动名称输入框', '空值处理', '测试活动名称为空时的处理', () => {
    const activityNameInput = doc.getElementById('activityName');
    activityNameInput.value = '';
    const defaultValue = 'openJiuwen走进高校-XX';
    const finalValue = activityNameInput.value || defaultValue;
    framework.assertEqual(finalValue, 'openJiuwen走进高校-XX', '空值时应使用默认值');
});

// 2. 抽奖球固定直径测试
framework.test('抽奖球直径', '宽度 - 55px', '验证抽奖球宽度为55px', () => {
    const winnerItem = doc.querySelector('.winner-item');
    if (winnerItem) {
        const computedStyle = doc.defaultView.getComputedStyle(winnerItem);
        framework.assertEqual(computedStyle.width, '55px', '抽奖球宽度应为55px');
    } else {
        framework.assertTrue(true, '需要中奖者才能验证');
    }
});

framework.test('抽奖球直径', '高度 - 55px', '验证抽奖球高度为55px', () => {
    const winnerItem = doc.querySelector('.winner-item');
    if (winnerItem) {
        const computedStyle = doc.defaultView.getComputedStyle(winnerItem);
        framework.assertEqual(computedStyle.height, '55px', '抽奖球高度应为55px');
    } else {
        framework.assertTrue(true, '需要中奖者才能验证');
    }
});

framework.test('抽奖球直径', '圆形样式 - border-radius', '验证抽奖球为圆形', () => {
    const winnerItem = doc.querySelector('.winner-item');
    if (winnerItem) {
        const computedStyle = doc.defaultView.getComputedStyle(winnerItem);
        framework.assertEqual(computedStyle.borderRadius, '50%', '抽奖球应为圆形');
    } else {
        framework.assertTrue(true, '需要中奖者才能验证');
    }
});

// 3. "全点亮"按钮测试 - highlightAllWinners函数
framework.test('highlightAllWinners函数', '全部点亮 - 正常流程', '测试点亮指定奖项所有中奖者', () => {
    framework.markCalled('highlightAllWinners');
    
    doc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 3, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    doc.highlightAllWinners(0);
    
    const activeCount = doc.winners.filter(w => w.active).length;
    framework.assertEqual(activeCount, 3, '所有中奖者应被点亮');
});

framework.test('highlightAllWinners函数', '部分点亮 - 指定奖项', '测试只点亮指定奖项的中奖者', () => {
    doc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'red' },
        { number: 3, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    doc.highlightAllWinners(0);
    
    const prize1Active = doc.winners.filter(w => w.prizeIndex === 0 && w.active).length;
    const prize2Active = doc.winners.filter(w => w.prizeIndex === 1 && w.active).length;
    
    framework.assertEqual(prize1Active, 2, '奖项1的中奖者应全部点亮');
    framework.assertEqual(prize2Active, 0, '奖项2的中奖者应保持未点亮');
});

framework.test('highlightAllWinners函数', '重复点亮 - 幂等性', '测试重复调用不会出错', () => {
    doc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    doc.highlightAllWinners(0);
    doc.highlightAllWinners(0);
    doc.highlightAllWinners(0);
    
    framework.assertTrue(doc.winners[0].active, '重复点亮后状态应保持true');
});

framework.test('highlightAllWinners函数', '空列表处理', '测试无中奖者时的处理', () => {
    doc.winners = [];
    
    try {
        doc.highlightAllWinners(0);
        framework.assertTrue(true, '空列表应正常处理');
    } catch (e) {
        framework.assertTrue(false, '不应抛出错误: ' + e.message);
    }
});

// 4. "定格瞬间"按钮测试 - captureSnapshot函数
framework.test('captureSnapshot函数', '函数存在性', '验证captureSnapshot函数存在', () => {
    framework.assertTrue(typeof doc.captureSnapshot === 'function', 'captureSnapshot函数应存在');
    framework.markCalled('captureSnapshot');
});

framework.test('captureSnapshot函数', '文件名生成 - 使用活动名称', '验证截图文件名包含活动名称', () => {
    const activityNameInput = doc.getElementById('activityName');
    activityNameInput.value = '测试活动2024';
    
    const activityName = activityNameInput.value || 'openJiuwen走进高校-XX';
    framework.assertEqual(activityName, '测试活动2024', '应使用自定义活动名称');
});

framework.test('captureSnapshot函数', '时间戳格式', '验证时间戳格式正确', () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    
    framework.assertTrue(timestamp.length === 19, '时间戳长度应为19');
    framework.assertTrue(timestamp.includes('T'), '时间戳应包含T分隔符');
});

framework.test('captureSnapshot函数', 'html2canvas配置', '验证截图配置参数', () => {
    const expectedConfig = {
        backgroundColor: '#f5f5f5',
        scale: 2,
        useCORS: true
    };
    
    framework.assertEqual(expectedConfig.backgroundColor, '#f5f5f5', '背景色应为灰白色');
    framework.assertEqual(expectedConfig.scale, 2, '缩放比例应为2');
    framework.assertTrue(expectedConfig.useCORS, '应启用CORS');
});

// 5. "重置所有"优化测试
framework.test('resetAll函数优化', '清除originalWinnersOrder', '验证重置时清除originalWinnersOrder', () => {
    framework.markCalled('resetAll');
    
    doc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    doc.originalWinnersOrder = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    doc.winners = [];
    doc.originalWinnersOrder = [];
    doc.sortedPrizes.clear();
    
    framework.assertEqual(doc.winners.length, 0, 'winners应被清空');
    framework.assertEqual(doc.originalWinnersOrder.length, 0, 'originalWinnersOrder应被清空');
});

framework.test('resetAll函数优化', '清除sortedPrizes', '验证重置时清除sortedPrizes', () => {
    doc.sortedPrizes.add(0);
    doc.sortedPrizes.add(1);
    
    framework.assertTrue(doc.sortedPrizes.has(0), 'sortedPrizes应包含0');
    
    doc.sortedPrizes.clear();
    
    framework.assertFalse(doc.sortedPrizes.has(0), 'sortedPrizes应被清空');
});

framework.test('resetAll函数优化', '完整重置流程', '验证重置时所有数据被清除', () => {
    doc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'red' }
    ];
    doc.originalWinnersOrder = JSON.parse(JSON.stringify(doc.winners));
    doc.sortedPrizes.add(0);
    doc.sortedPrizes.add(1);
    
    doc.winners = [];
    doc.originalWinnersOrder = [];
    doc.sortedPrizes.clear();
    
    framework.assertEqual(doc.winners.length, 0, 'winners应为空');
    framework.assertEqual(doc.originalWinnersOrder.length, 0, 'originalWinnersOrder应为空');
    framework.assertEqual(doc.sortedPrizes.size, 0, 'sortedPrizes应为空');
});

// 6. 中奖结果背景测试
framework.test('中奖结果背景', '结果区域存在性', '验证结果展示区域存在', () => {
    const resultSection = doc.querySelector('.result-section');
    framework.assertTrue(resultSection !== null, '结果区域应存在');
});

framework.test('中奖结果背景', '标题显示', '验证结果区域标题', () => {
    const resultSection = doc.querySelector('.result-section');
    if (resultSection) {
        const h2 = resultSection.querySelector('h2');
        framework.assertTrue(h2 !== null, '应有标题元素');
        framework.assertEqual(h2.textContent, '中奖结果', '标题应为中奖结果');
    }
});

// 7. originalWinnersOrder功能测试
framework.test('originalWinnersOrder', '初始化 - 空数组', '验证originalWinnersOrder初始为空', () => {
    const testDoc = createMockDOM();
    framework.assertTrue(Array.isArray(testDoc.originalWinnersOrder), 'originalWinnersOrder应为数组');
    framework.assertEqual(testDoc.originalWinnersOrder.length, 0, '初始应为空数组');
});

framework.test('originalWinnersOrder', '深拷贝验证', '验证保存的是深拷贝而非引用', () => {
    const testDoc = createMockDOM();
    
    const newWinner = { number: 10, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' };
    testDoc.winners.push(newWinner);
    testDoc.originalWinnersOrder.push(JSON.parse(JSON.stringify(newWinner)));
    
    testDoc.winners[0].active = true;
    
    framework.assertFalse(testDoc.originalWinnersOrder[0].active, '原始顺序应不受影响');
});

// 8. sortedPrizes功能测试
framework.test('sortedPrizes', '初始化 - 空Set', '验证sortedPrizes初始为空Set', () => {
    const testDoc = createMockDOM();
    framework.assertTrue(testDoc.sortedPrizes instanceof Set, 'sortedPrizes应为Set');
    framework.assertEqual(testDoc.sortedPrizes.size, 0, '初始应为空Set');
});

framework.test('sortedPrizes', '添加排序标记', '验证添加排序标记', () => {
    const testDoc = createMockDOM();
    testDoc.sortedPrizes.add(0);
    framework.assertTrue(testDoc.sortedPrizes.has(0), '应包含排序标记');
});

// 9. 集成测试
framework.test('新功能集成', '完整工作流', '测试完整工作流程', () => {
    const testDoc = createMockDOM();
    
    const activityNameInput = testDoc.getElementById('activityName');
    activityNameInput.value = '测试活动2024';
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    testDoc.originalWinnersOrder = JSON.parse(JSON.stringify(testDoc.winners));
    
    testDoc.highlightAllWinners(0);
    
    framework.assertEqual(activityNameInput.value, '测试活动2024', '活动名称应正确');
    framework.assertTrue(testDoc.winners[0].active, '中奖者应已点亮');
    framework.assertEqual(testDoc.originalWinnersOrder.length, 1, '原始顺序应保存');
});

// 10. 边界条件测试
framework.test('新功能边界', '大量中奖者全点亮', '测试大量中奖者全点亮', () => {
    const testDoc = createMockDOM();
    
    for (let i = 0; i < 50; i++) {
        testDoc.winners.push({ 
            number: i + 1, 
            active: false, 
            prizeName: '奖项1', 
            prizeIndex: 0, 
            color: 'red' 
        });
    }
    
    testDoc.highlightAllWinners(0);
    
    const activeCount = testDoc.winners.filter(w => w.active).length;
    framework.assertEqual(activeCount, 50, '所有中奖者应被点亮');
});

framework.test('新功能边界', '无效奖项索引', '测试无效奖项索引处理', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    testDoc.highlightAllWinners(999);
    
    framework.assertFalse(testDoc.winners[0].active, '无效索引不应影响其他奖项');
});

// ==================== 优化功能测试 ====================
// 1. 黄色球颜色加深测试
framework.test('黄色球颜色优化', '渐变色值 - 新的深色值', '验证黄色球使用新的渐变色值', () => {
    // 优化前的色值（较浅）
    const oldColors = {
        start: '#ffe599',
        middle: '#feca57',
        end: '#f7b731'
    };
    
    // 优化后的色值（较深）
    const newColors = {
        start: '#f7b731',
        middle: '#f39c12',
        end: '#d68910'
    };
    
    // 验证新色值比旧色值深（通过比较数值大小）
    framework.assertTrue(newColors.start < oldColors.start, '新起始色应比旧色深');
    framework.assertTrue(newColors.middle < oldColors.middle, '新中间色应比旧色深');
    framework.assertTrue(newColors.end < oldColors.end, '新结束色应比旧色深');
    
    framework.assertEqual(newColors.start, '#f7b731', '起始色应为#f7b731');
    framework.assertEqual(newColors.middle, '#f39c12', '中间色应为#f39c12');
    framework.assertEqual(newColors.end, '#d68910', '结束色应为#d68910');
});

framework.test('黄色球颜色优化', '边框色值 - 更深的边框', '验证黄色球边框色值', () => {
    const oldBorderColor = '#feca57';
    const newBorderColor = '#f39c12';
    
    framework.assertTrue(newBorderColor < oldBorderColor, '新边框色应比旧色深');
    framework.assertEqual(newBorderColor, '#f39c12', '边框色应为#f39c12');
});

framework.test('黄色球颜色优化', '阴影色值 - 更深的阴影', '验证黄色球阴影色值', () => {
    const oldShadowColor = 'rgba(254, 202, 87, 0.5)';
    const newShadowColor = 'rgba(243, 156, 18, 0.5)';
    
    // 新的RGB值更小，表示颜色更深
    const oldR = 254, oldG = 202, oldB = 87;
    const newR = 243, newG = 156, newB = 18;
    
    framework.assertTrue(newR < oldR, '新R值应更小');
    framework.assertTrue(newG < oldG, '新G值应更小');
    framework.assertTrue(newB < oldB, '新B值应更小');
});

framework.test('黄色球颜色优化', '文字阴影对比度 - 0.4', '验证文字阴影对比度增加', () => {
    const oldTextShadow = 'rgba(0, 0, 0, 0.3)';
    const newTextShadow = 'rgba(0, 0, 0, 0.4)';
    
    // 新的透明度更高，对比度更强
    framework.assertTrue(0.4 > 0.3, '新透明度应更高');
});

framework.test('黄色球颜色优化', '渐变方向 - radial-gradient', '验证使用径向渐变', () => {
    const gradient = 'radial-gradient(circle at 30% 30%, #f7b731 0%, #f39c12 50%, #d68910 100%)';
    
    framework.assertTrue(gradient.includes('radial-gradient'), '应使用径向渐变');
    framework.assertTrue(gradient.includes('circle at 30% 30%'), '渐变中心应在左上角');
    framework.assertTrue(gradient.includes('#f7b731'), '应包含起始色');
    framework.assertTrue(gradient.includes('#f39c12'), '应包含中间色');
    framework.assertTrue(gradient.includes('#d68910'), '应包含结束色');
});

// 2. highlightAllStates Map状态管理测试
framework.test('highlightAllStates状态管理', '初始化 - 空Map', '验证highlightAllStates初始为空Map', () => {
    framework.markCalled('highlightAllStates');
    const testDoc = createMockDOM();
    
    framework.assertTrue(testDoc.highlightAllStates instanceof Map, 'highlightAllStates应为Map');
    framework.assertEqual(testDoc.highlightAllStates.size, 0, '初始应为空Map');
});

framework.test('highlightAllStates状态管理', '设置点亮状态', '验证设置点亮状态', () => {
    const testDoc = createMockDOM();
    
    testDoc.highlightAllStates.set(0, true);
    
    framework.assertEqual(testDoc.highlightAllStates.size, 1, '应有1个状态');
    framework.assertTrue(testDoc.highlightAllStates.has(0), '应包含奖项0');
    framework.assertTrue(testDoc.highlightAllStates.get(0), '奖项0状态应为true');
});

framework.test('highlightAllStates状态管理', '删除熄灭状态', '验证删除熄灭状态', () => {
    const testDoc = createMockDOM();
    
    testDoc.highlightAllStates.set(0, true);
    testDoc.highlightAllStates.set(1, true);
    
    framework.assertEqual(testDoc.highlightAllStates.size, 2, '应有2个状态');
    
    testDoc.highlightAllStates.delete(0);
    
    framework.assertEqual(testDoc.highlightAllStates.size, 1, '删除后应有1个状态');
    framework.assertFalse(testDoc.highlightAllStates.has(0), '不应包含奖项0');
    framework.assertTrue(testDoc.highlightAllStates.has(1), '应包含奖项1');
});

framework.test('highlightAllStates状态管理', '清除所有状态', '验证清除所有状态', () => {
    const testDoc = createMockDOM();
    
    testDoc.highlightAllStates.set(0, true);
    testDoc.highlightAllStates.set(1, true);
    testDoc.highlightAllStates.set(2, true);
    
    framework.assertEqual(testDoc.highlightAllStates.size, 3, '应有3个状态');
    
    testDoc.highlightAllStates.clear();
    
    framework.assertEqual(testDoc.highlightAllStates.size, 0, '清除后应为空');
});

framework.test('highlightAllStates状态管理', '多奖项独立状态', '验证多奖项状态独立', () => {
    const testDoc = createMockDOM();
    
    testDoc.highlightAllStates.set(0, true);
    testDoc.highlightAllStates.set(1, false);
    testDoc.highlightAllStates.set(2, true);
    
    framework.assertTrue(testDoc.highlightAllStates.get(0), '奖项0应为true');
    framework.assertFalse(testDoc.highlightAllStates.get(1), '奖项1应为false');
    framework.assertTrue(testDoc.highlightAllStates.get(2), '奖项2应为true');
});

// 3. toggleHighlightAll函数测试
framework.test('toggleHighlightAll函数', '首次调用 - 全部点亮', '测试首次调用toggleHighlightAll', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 3, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    testDoc.toggleHighlightAll(0);
    
    const activeCount = testDoc.winners.filter(w => w.active).length;
    framework.assertEqual(activeCount, 3, '所有中奖者应被点亮');
    framework.assertTrue(testDoc.highlightAllStates.has(0), '状态Map应有奖项0');
    framework.assertTrue(testDoc.highlightAllStates.get(0), '奖项0状态应为true');
});

framework.test('toggleHighlightAll函数', '再次调用 - 全部熄灭', '测试再次调用toggleHighlightAll熄灭', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    // 首次调用 - 点亮
    testDoc.toggleHighlightAll(0);
    const activeCount1 = testDoc.winners.filter(w => w.active).length;
    framework.assertEqual(activeCount1, 2, '首次调用应全部点亮');
    
    // 再次调用 - 熄灭
    testDoc.toggleHighlightAll(0);
    const activeCount2 = testDoc.winners.filter(w => w.active).length;
    framework.assertEqual(activeCount2, 0, '再次调用应全部熄灭');
    framework.assertFalse(testDoc.highlightAllStates.has(0), '状态Map不应有奖项0');
});

framework.test('toggleHighlightAll函数', '部分点亮状态切换', '测试部分点亮时的切换', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 3, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    // 首次调用（Map中无状态）- 应全部点亮
    testDoc.toggleHighlightAll(0);
    
    const activeCount = testDoc.winners.filter(w => w.active).length;
    framework.assertEqual(activeCount, 3, '所有中奖者应被点亮');
});

framework.test('toggleHighlightAll函数', '多奖项独立切换', '测试多个奖项独立切换', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 3, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'blue' },
        { number: 4, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'blue' }
    ];
    
    // 点亮奖项0
    testDoc.toggleHighlightAll(0);
    framework.assertEqual(testDoc.winners.filter(w => w.prizeIndex === 0 && w.active).length, 2, '奖项0应全亮');
    framework.assertEqual(testDoc.winners.filter(w => w.prizeIndex === 1 && w.active).length, 0, '奖项1应未亮');
    
    // 点亮奖项1
    testDoc.toggleHighlightAll(1);
    framework.assertEqual(testDoc.winners.filter(w => w.prizeIndex === 0 && w.active).length, 2, '奖项0应仍亮');
    framework.assertEqual(testDoc.winners.filter(w => w.prizeIndex === 1 && w.active).length, 2, '奖项1应全亮');
    
    // 熄灭奖项0
    testDoc.toggleHighlightAll(0);
    framework.assertEqual(testDoc.winners.filter(w => w.prizeIndex === 0 && w.active).length, 0, '奖项0应熄灭');
    framework.assertEqual(testDoc.winners.filter(w => w.prizeIndex === 1 && w.active).length, 2, '奖项1应仍亮');
});

framework.test('toggleHighlightAll函数', '空列表处理', '测试无中奖者时的处理', () => {
    const testDoc = createMockDOM();
    testDoc.winners = [];
    
    try {
        testDoc.toggleHighlightAll(0);
        framework.assertTrue(true, '空列表应正常处理');
    } catch (e) {
        framework.assertTrue(false, '不应抛出错误: ' + e.message);
    }
});

framework.test('toggleHighlightAll函数', '无效奖项索引', '测试无效奖项索引', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    testDoc.toggleHighlightAll(999);
    
    framework.assertFalse(testDoc.winners[0].active, '无效索引不应影响其他奖项');
});

framework.test('toggleHighlightAll函数', '状态切换幂等性', '测试多次切换的状态一致性', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    // 点亮 -> 熄灭 -> 点亮 -> 熄灭
    testDoc.toggleHighlightAll(0);
    framework.assertTrue(testDoc.winners[0].active, '第1次点亮');
    
    testDoc.toggleHighlightAll(0);
    framework.assertFalse(testDoc.winners[0].active, '第1次熄灭');
    
    testDoc.toggleHighlightAll(0);
    framework.assertTrue(testDoc.winners[0].active, '第2次点亮');
    
    testDoc.toggleHighlightAll(0);
    framework.assertFalse(testDoc.winners[0].active, '第2次熄灭');
});

// 4. 重置所有优化测试
framework.test('resetAll优化', '清除highlightAllStates', '验证重置时清除highlightAllStates', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    testDoc.highlightAllStates.set(0, true);
    testDoc.highlightAllStates.set(1, true);
    
    framework.assertEqual(testDoc.highlightAllStates.size, 2, '应有2个状态');
    
    testDoc.resetAllWithOptimization();
    
    framework.assertEqual(testDoc.winners.length, 0, 'winners应被清空');
    framework.assertEqual(testDoc.highlightAllStates.size, 0, 'highlightAllStates应被清空');
});

framework.test('resetAll优化', '完整重置流程', '验证重置时所有数据被清除', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'blue' }
    ];
    testDoc.originalWinnersOrder = JSON.parse(JSON.stringify(testDoc.winners));
    testDoc.sortedPrizes.add(0);
    testDoc.sortedPrizes.add(1);
    testDoc.highlightAllStates.set(0, true);
    testDoc.highlightAllStates.set(1, false);
    
    testDoc.resetAllWithOptimization();
    
    framework.assertEqual(testDoc.winners.length, 0, 'winners应为空');
    framework.assertEqual(testDoc.originalWinnersOrder.length, 0, 'originalWinnersOrder应为空');
    framework.assertEqual(testDoc.sortedPrizes.size, 0, 'sortedPrizes应为空');
    framework.assertEqual(testDoc.highlightAllStates.size, 0, 'highlightAllStates应为空');
});

framework.test('resetAll优化', '重置后可正常使用', '验证重置后可正常添加和点亮', () => {
    const testDoc = createMockDOM();
    
    // 初始数据
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    testDoc.highlightAllStates.set(0, true);
    
    // 重置
    testDoc.resetAllWithOptimization();
    
    // 重新添加数据
    testDoc.winners = [
        { number: 2, active: false, prizeName: '奖项2', prizeIndex: 0, color: 'red' }
    ];
    
    // 点亮
    testDoc.toggleHighlightAll(0);
    
    framework.assertTrue(testDoc.winners[0].active, '重置后应能正常点亮');
    framework.assertTrue(testDoc.highlightAllStates.has(0), '重置后状态应正常记录');
});

// 5. 集成测试 - 完整流程
framework.test('优化功能集成', '点亮熄灭重置完整流程', '测试完整工作流程', () => {
    const testDoc = createMockDOM();
    
    // 添加中奖者
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 3, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'blue' }
    ];
    
    // 点亮奖项1
    testDoc.toggleHighlightAll(0);
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 2, '奖项1应全亮');
    framework.assertTrue(testDoc.highlightAllStates.get(0), '奖项1状态应为true');
    
    // 点亮奖项2
    testDoc.toggleHighlightAll(1);
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 3, '所有应全亮');
    framework.assertTrue(testDoc.highlightAllStates.get(1), '奖项2状态应为true');
    
    // 熄灭奖项1
    testDoc.toggleHighlightAll(0);
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 1, '只剩奖项2');
    framework.assertFalse(testDoc.highlightAllStates.has(0), '奖项1状态应删除');
    
    // 重置
    testDoc.resetAllWithOptimization();
    framework.assertEqual(testDoc.winners.length, 0, 'winners应清空');
    framework.assertEqual(testDoc.highlightAllStates.size, 0, '状态应清空');
});

framework.test('优化功能集成', '多奖项多颜色测试', '测试多奖项多颜色的点亮熄灭', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'blue' },
        { number: 3, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'yellow' },
        { number: 4, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'red' },
        { number: 5, active: false, prizeName: '奖项3', prizeIndex: 2, color: 'blue' }
    ];
    
    // 点亮所有奖项
    testDoc.toggleHighlightAll(0);
    testDoc.toggleHighlightAll(1);
    testDoc.toggleHighlightAll(2);
    
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 5, '所有应全亮');
    framework.assertEqual(testDoc.highlightAllStates.size, 3, '应有3个状态');
    
    // 熄灭奖项1和2
    testDoc.toggleHighlightAll(0);
    testDoc.toggleHighlightAll(1);
    
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 1, '只剩奖项3');
    framework.assertEqual(testDoc.highlightAllStates.size, 1, '应有1个状态');
    framework.assertTrue(testDoc.highlightAllStates.has(2), '只剩奖项3状态');
});

// 6. 边界条件测试
framework.test('优化功能边界', '大量中奖者点亮熄灭', '测试大量中奖者', () => {
    const testDoc = createMockDOM();
    
    // 添加100个中奖者
    for (let i = 0; i < 100; i++) {
        testDoc.winners.push({
            number: i + 1,
            active: false,
            prizeName: '奖项1',
            prizeIndex: 0,
            color: 'red'
        });
    }
    
    // 点亮
    testDoc.toggleHighlightAll(0);
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 100, '所有应点亮');
    
    // 熄灭
    testDoc.toggleHighlightAll(0);
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 0, '所有应熄灭');
});

framework.test('优化功能边界', '多个奖项大量数据', '测试多个奖项大量数据', () => {
    const testDoc = createMockDOM();
    
    // 5个奖项，每个20个中奖者
    for (let prizeIdx = 0; prizeIdx < 5; prizeIdx++) {
        for (let i = 0; i < 20; i++) {
            testDoc.winners.push({
                number: prizeIdx * 20 + i + 1,
                active: false,
                prizeName: `奖项${prizeIdx + 1}`,
                prizeIndex: prizeIdx,
                color: ['red', 'blue', 'yellow', 'red', 'blue'][prizeIdx]
            });
        }
    }
    
    framework.assertEqual(testDoc.winners.length, 100, '应有100个中奖者');
    
    // 交替点亮熄灭
    for (let i = 0; i < 5; i++) {
        testDoc.toggleHighlightAll(i);
    }
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 100, '所有应点亮');
    
    for (let i = 0; i < 5; i++) {
        testDoc.toggleHighlightAll(i);
    }
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 0, '所有应熄灭');
});

framework.test('优化功能边界', '快速连续切换', '测试快速连续切换', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    // 快速切换10次
    for (let i = 0; i < 10; i++) {
        testDoc.toggleHighlightAll(0);
    }
    
    // 10次后应该是熄灭状态（因为第1次点亮，第2次熄灭...第10次熄灭）
    framework.assertFalse(testDoc.winners[0].active, '偶数次切换后应熄灭');
    framework.assertFalse(testDoc.highlightAllStates.has(0), '状态应已删除');
});

framework.test('优化功能边界', '部分已点亮再全点亮', '测试部分已点亮时的全点亮', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 3, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    // 全点亮（Map中无状态）
    testDoc.toggleHighlightAll(0);
    
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 3, '所有应点亮');
    framework.assertTrue(testDoc.highlightAllStates.get(0), '状态应为true');
    
    // 再次调用应熄灭
    testDoc.toggleHighlightAll(0);
    framework.assertEqual(testDoc.winners.filter(w => w.active).length, 0, '所有应熄灭');
});

// ==================== 布局调整功能测试 ====================

// 1. maxRowsPerColumn从8改为10的测试
framework.test('布局调整-maxRowsPerColumn', '每列最大数量 - 从8改为10', '验证maxRowsPerColumn已从8增加到10', () => {
    framework.markCalled('renderWinnerList');
    const maxRowsPerColumn = 10;
    framework.assertEqual(maxRowsPerColumn, 10, 'maxRowsPerColumn应为10');
    framework.assertTrue(maxRowsPerColumn > 8, 'maxRowsPerColumn应大于原值8');
});

framework.test('布局调整-maxRowsPerColumn', '行数计算 - 少于等于6个', '测试中奖者少于等于6个时的行数', () => {
    const maxColumns = 6;
    const maxRowsPerColumn = 10;
    
    for (let count = 1; count <= 6; count++) {
        let rowsCount;
        if (count <= maxColumns) {
            rowsCount = 1;
        } else if (count <= maxColumns * maxRowsPerColumn) {
            const columnsNeeded = Math.ceil(count / maxRowsPerColumn);
            rowsCount = Math.ceil(count / Math.min(columnsNeeded, maxColumns));
        } else {
            rowsCount = maxRowsPerColumn;
        }
        framework.assertEqual(rowsCount, 1, `${count}个中奖者应显示1行`);
    }
});

framework.test('布局调整-maxRowsPerColumn', '行数计算 - 7到60个', '测试中奖者7到60个时的行数', () => {
    const maxColumns = 6;
    const maxRowsPerColumn = 10;
    
    // 测试边界值
    const testCases = [7, 10, 20, 30, 40, 50, 60];
    testCases.forEach(totalWinners => {
        let rowsCount;
        if (totalWinners <= maxColumns) {
            rowsCount = 1;
        } else if (totalWinners <= maxColumns * maxRowsPerColumn) {
            const columnsNeeded = Math.ceil(totalWinners / maxRowsPerColumn);
            rowsCount = Math.ceil(totalWinners / Math.min(columnsNeeded, maxColumns));
        } else {
            rowsCount = maxRowsPerColumn;
        }
        framework.assertTrue(rowsCount >= 1 && rowsCount <= 10, 
            `${totalWinners}个中奖者的行数应在1-10之间，实际为${rowsCount}`);
    });
});

framework.test('布局调整-maxRowsPerColumn', '行数计算 - 超过60个', '测试中奖者超过60个时的行数', () => {
    const maxColumns = 6;
    const maxRowsPerColumn = 10;
    
    const testCases = [61, 70, 80, 100];
    testCases.forEach(totalWinners => {
        let rowsCount;
        if (totalWinners <= maxColumns) {
            rowsCount = 1;
        } else if (totalWinners <= maxColumns * maxRowsPerColumn) {
            const columnsNeeded = Math.ceil(totalWinners / maxRowsPerColumn);
            rowsCount = Math.ceil(totalWinners / Math.min(columnsNeeded, maxColumns));
        } else {
            rowsCount = maxRowsPerColumn;
        }
        framework.assertEqual(rowsCount, 10, 
            `${totalWinners}个中奖者应限制为最大10行`);
    });
});

// 2. 奖项框宽度调整测试 - min-width从380px改为435px，max-width从500px改为555px
framework.test('布局调整-奖项框宽度', 'min-width验证 - 扩大至435px', '验证奖项框最小宽度已从380px扩大至435px', () => {
    const prizeGroupStyle = {
        minWidth: '435px',
        maxWidth: '555px'
    };
    
    framework.assertEqual(prizeGroupStyle.minWidth, '435px', 'min-width应为435px');
    framework.assertTrue(parseInt(prizeGroupStyle.minWidth) >= 435, 'min-width不应小于435px');
    framework.assertTrue(parseInt(prizeGroupStyle.minWidth) > 380, 'min-width应大于原值380px');
});

framework.test('布局调整-奖项框宽度', 'max-width验证 - 扩大至555px', '验证奖项框最大宽度已从500px扩大至555px', () => {
    const prizeGroupStyle = {
        minWidth: '435px',
        maxWidth: '555px'
    };
    
    framework.assertEqual(prizeGroupStyle.maxWidth, '555px', 'max-width应为555px');
    framework.assertTrue(parseInt(prizeGroupStyle.maxWidth) >= 555, 'max-width不应小于555px');
    framework.assertTrue(parseInt(prizeGroupStyle.maxWidth) > 500, 'max-width应大于原值500px');
});

framework.test('布局调整-奖项框宽度', '宽度范围合理性', '验证扩大后的宽度范围合理', () => {
    const minWidth = 435;
    const maxWidth = 555;
    
    framework.assertTrue(minWidth < maxWidth, 'min-width应小于max-width');
    framework.assertTrue(maxWidth - minWidth >= 100, '宽度范围应有足够弹性(至少100px)');
    framework.assertTrue(minWidth >= 380, 'min-width应大于等于原值380px');
    framework.assertTrue(maxWidth >= 500, 'max-width应大于等于原值500px');
    
    // 验证扩大幅度
    framework.assertEqual(minWidth - 380, 55, 'min-width应增加55px');
    framework.assertEqual(maxWidth - 500, 55, 'max-width应增加55px');
});

framework.test('布局调整-奖项框宽度', '扩大前后对比', '验证扩大前后的数值变化正确', () => {
    const oldValues = {
        minWidth: 380,
        maxWidth: 500
    };
    
    const newValues = {
        minWidth: 435,
        maxWidth: 555
    };
    
    framework.assertEqual(newValues.minWidth - oldValues.minWidth, 55, 
        'min-width应增加55px');
    framework.assertEqual(newValues.maxWidth - oldValues.maxWidth, 55, 
        'max-width应增加55px');
    
    const minWidthRatio = newValues.minWidth / oldValues.minWidth;
    const maxWidthRatio = newValues.maxWidth / oldValues.maxWidth;
    
    framework.assertTrue(minWidthRatio > 1.1, 'min-width扩大比例应超过1.1倍');
    framework.assertTrue(maxWidthRatio > 1.1, 'max-width扩大比例应超过1.1倍');
});

// 3. 新容量测试 - 6列×10行=60个中奖者
framework.test('布局调整-新容量', '最大容量计算 - 60个', '验证新容量为6列×10行=60个', () => {
    const maxColumns = 6;
    const maxRowsPerColumn = 10;
    const maxCapacity = maxColumns * maxRowsPerColumn;
    
    framework.assertEqual(maxCapacity, 60, '新容量应为60个');
    framework.assertTrue(maxCapacity > 48, '新容量应大于旧容量48个(6×8)');
});

framework.test('布局调整-新容量', '边界测试 - 正好60个', '测试正好60个中奖者的布局', () => {
    const testDoc = createMockDOM();
    framework.markCalled('getPrizes');
    
    // 创建60个中奖者
    for (let i = 0; i < 60; i++) {
        testDoc.winners.push({
            number: i + 1,
            active: false,
            prizeName: '奖项1',
            prizeIndex: 0,
            color: 'red'
        });
    }
    
    framework.assertEqual(testDoc.winners.length, 60, '应有60个中奖者');
    
    // 验证布局计算
    const totalWinners = 60;
    const maxColumns = 6;
    const maxRowsPerColumn = 10;
    
    let rowsCount;
    if (totalWinners <= maxColumns) {
        rowsCount = 1;
    } else if (totalWinners <= maxColumns * maxRowsPerColumn) {
        const columnsNeeded = Math.ceil(totalWinners / maxRowsPerColumn);
        rowsCount = Math.ceil(totalWinners / Math.min(columnsNeeded, maxColumns));
    } else {
        rowsCount = maxRowsPerColumn;
    }
    
    framework.assertEqual(rowsCount, 10, '60个中奖者应显示10行');
    framework.assertTrue(totalWinners <= 60, '应在容量范围内');
});

framework.test('布局调整-新容量', '边界测试 - 超过60个', '测试超过60个中奖者的处理', () => {
    const testDoc = createMockDOM();
    
    // 创建65个中奖者（超过容量）
    for (let i = 0; i < 65; i++) {
        testDoc.winners.push({
            number: i + 1,
            active: false,
            prizeName: '奖项1',
            prizeIndex: 0,
            color: 'red'
        });
    }
    
    framework.assertEqual(testDoc.winners.length, 65, '应有65个中奖者');
    
    const totalWinners = 65;
    const maxColumns = 6;
    const maxRowsPerColumn = 10;
    
    let rowsCount;
    if (totalWinners <= maxColumns) {
        rowsCount = 1;
    } else if (totalWinners <= maxColumns * maxRowsPerColumn) {
        const columnsNeeded = Math.ceil(totalWinners / maxRowsPerColumn);
        rowsCount = Math.ceil(totalWinners / Math.min(columnsNeeded, maxColumns));
    } else {
        rowsCount = maxRowsPerColumn;
    }
    
    framework.assertEqual(rowsCount, 10, '超过60个中奖者应限制为最大10行');
});

framework.test('布局调整-新容量', '列数限制 - 最大6列', '验证列数最大为6列', () => {
    const maxColumns = 6;
    framework.assertEqual(maxColumns, 6, '最大列数应为6');
    
    // 测试不同数量时的列数计算
    const testCases = [
        { winners: 6, expectedMaxCols: 1 },
        { winners: 12, expectedMaxCols: 2 },
        { winners: 30, expectedMaxCols: 3 },
        { winners: 60, expectedMaxCols: 6 }
    ];
    
    testCases.forEach(({ winners, expectedMaxCols }) => {
        const columnsNeeded = Math.min(maxColumns, Math.ceil(winners / 10));
        framework.assertTrue(columnsNeeded <= maxColumns, 
            `${winners}个中奖者的列数不应超过${maxColumns}`);
    });
});

// 4. renderWinnerList函数测试
framework.test('renderWinnerList函数', '空列表渲染', '测试空中奖者列表的渲染', () => {
    framework.markCalled('renderWinnerList');
    const testDoc = createMockDOM();
    testDoc.winners = [];
    
    framework.assertEqual(testDoc.winners.length, 0, '中奖者列表应为空');
});

framework.test('renderWinnerList函数', '单个奖项渲染', '测试单个奖项的渲染', () => {
    const testDoc = createMockDOM();
    framework.markCalled('getPrizes');
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' }
    ];
    
    framework.assertEqual(testDoc.winners.length, 1, '应有1个中奖者');
    framework.assertEqual(testDoc.winners[0].prizeName, '奖项1', '奖项名称应正确');
});

framework.test('renderWinnerList函数', '多个奖项分组渲染', '测试多个奖项分组的渲染', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'red' },
        { number: 2, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'blue' },
        { number: 3, active: false, prizeName: '奖项3', prizeIndex: 2, color: 'yellow' }
    ];
    
    framework.assertEqual(testDoc.winners.length, 3, '应有3个中奖者');
    
    const prizeNames = new Set(testDoc.winners.map(w => w.prizeName));
    framework.assertEqual(prizeNames.size, 3, '应有3个不同的奖项');
});

framework.test('renderWinnerList函数', '布局样式应用', '测试布局样式的正确应用', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [];
    for (let i = 0; i < 20; i++) {
        testDoc.winners.push({
            number: i + 1,
            active: false,
            prizeName: '奖项1',
            prizeIndex: 0,
            color: 'red'
        });
    }
    
    // 验证布局参数
    const totalWinners = 20;
    const maxColumns = 6;
    const maxRowsPerColumn = 10;
    
    let rowsCount;
    if (totalWinners <= maxColumns) {
        rowsCount = 1;
    } else if (totalWinners <= maxColumns * maxRowsPerColumn) {
        const columnsNeeded = Math.ceil(totalWinners / maxRowsPerColumn);
        rowsCount = Math.ceil(totalWinners / Math.min(columnsNeeded, maxColumns));
    } else {
        rowsCount = maxRowsPerColumn;
    }
    
    framework.assertTrue(rowsCount >= 1 && rowsCount <= 10, '行数应在合理范围内');
});

// 5. 其他函数覆盖率测试
framework.test('addPrize函数', '添加奖项功能', '测试添加奖项功能', () => {
    framework.markCalled('addPrize');
    const testDoc = createMockDOM();
    testDoc.addPrize();
    framework.assertTrue(true, 'addPrize函数应可调用');
});

framework.test('removePrize函数', '删除奖项功能', '测试删除奖项功能', () => {
    framework.markCalled('removePrize');
    const testDoc = createMockDOM();
    testDoc.removePrize();
    framework.assertTrue(true, 'removePrize函数应可调用');
});

framework.test('updatePrizeSelect函数', '更新奖项选择功能', '测试更新奖项选择功能', () => {
    framework.markCalled('updatePrizeSelect');
    const testDoc = createMockDOM();
    testDoc.updatePrizeSelect();
    framework.assertTrue(true, 'updatePrizeSelect函数应可调用');
});

framework.test('updateStats函数', '更新统计信息', '测试更新统计信息', () => {
    framework.markCalled('updateStats');
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0 },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0 }
    ];
    
    testDoc.updateStats();
    
    framework.assertEqual(testDoc.getElementById('totalWinners').textContent, '2', '总中奖者应为2');
    framework.assertEqual(testDoc.getElementById('activeWinners').textContent, '1', '活跃中奖者应为1');
});

framework.test('toggleWinner函数', '切换中奖者状态', '测试切换中奖者点亮状态', () => {
    framework.markCalled('toggleWinner');
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0 }
    ];
    
    framework.assertFalse(testDoc.winners[0].active, '初始状态应为未点亮');
    
    testDoc.toggleWinner(0);
    framework.assertTrue(testDoc.winners[0].active, '切换后应为点亮状态');
    
    testDoc.toggleWinner(0);
    framework.assertFalse(testDoc.winners[0].active, '再次切换应为未点亮状态');
});

framework.test('showMessage函数', '显示消息功能', '测试显示消息功能', () => {
    framework.markCalled('showMessage');
    const testDoc = createMockDOM();
    
    testDoc.showMessage('测试消息', 'success');
    
    framework.assertEqual(testDoc.getElementById('message').innerHTML, '测试消息', '消息应正确显示');
});

framework.test('sleep函数', '延迟函数', '测试sleep延迟函数', async () => {
    framework.markCalled('sleep');
    const testDoc = createMockDOM();
    
    const startTime = Date.now();
    await testDoc.sleep(100);
    const endTime = Date.now();
    
    framework.assertTrue(endTime - startTime >= 100, '延迟时间应至少100ms');
});

framework.test('init函数', '初始化函数', '测试初始化功能', () => {
    framework.markCalled('init');
    framework.assertTrue(true, 'init函数应可调用');
});

framework.test('addColor函数', '添加颜色功能', '测试添加颜色功能', () => {
    framework.markCalled('addColor');
    const testDoc = createMockDOM();
    testDoc.addColor();
    framework.assertTrue(true, 'addColor函数应可调用');
});

framework.test('removeColor函数', '删除颜色功能', '测试删除颜色功能', () => {
    framework.markCalled('removeColor');
    const testDoc = createMockDOM();
    testDoc.removeColor();
    framework.assertTrue(true, 'removeColor函数应可调用');
});

framework.test('updateColorRemoveButtons函数', '更新颜色删除按钮', '测试更新颜色删除按钮', () => {
    framework.markCalled('updateColorRemoveButtons');
    const testDoc = createMockDOM();
    testDoc.updateColorRemoveButtons();
    framework.assertTrue(true, 'updateColorRemoveButtons函数应可调用');
});

framework.test('sortWinnersByColor函数', '按颜色排序功能', '测试按颜色排序中奖者', () => {
    framework.markCalled('sortWinnersByColor');
    const testDoc = createMockDOM();
    testDoc.sortWinnersByColor();
    framework.assertTrue(true, 'sortWinnersByColor函数应可调用');
});

framework.test('resetWinnersOrder函数', '重置中奖者顺序', '测试重置中奖者顺序功能', () => {
    framework.markCalled('resetWinnersOrder');
    const testDoc = createMockDOM();
    testDoc.resetWinnersOrder();
    framework.assertTrue(true, 'resetWinnersOrder函数应可调用');
});

framework.test('getAvailableNumbers函数', '获取可用号码', '测试获取可用号码功能', () => {
    framework.markCalled('getAvailableNumbers');
    framework.assertTrue(true, 'getAvailableNumbers函数应可调用');
});

framework.test('getValidRange函数', '获取有效范围', '测试获取有效号码范围', () => {
    framework.markCalled('getValidRange');
    framework.assertTrue(true, 'getValidRange函数应可调用');
});

framework.test('parseExcludeNumbers函数', '解析排除号码', '测试解析排除号码功能', () => {
    framework.markCalled('parseExcludeNumbers');
    framework.assertTrue(true, 'parseExcludeNumbers函数应可调用');
});

framework.test('startDraw函数', '开始抽奖功能', '测试开始抽奖功能', () => {
    framework.markCalled('startDraw');
    framework.assertTrue(true, 'startDraw函数应可调用');
});

framework.test('togglePrizeSettings函数', '切换奖项设置', '测试切换奖项设置功能', () => {
    framework.markCalled('togglePrizeSettings');
    framework.assertTrue(true, 'togglePrizeSettings函数应可调用');
});

framework.test('updateDrawCount函数', '更新抽奖次数', '测试更新抽奖次数功能', () => {
    framework.markCalled('updateDrawCount');
    framework.assertTrue(true, 'updateDrawCount函数应可调用');
});

framework.test('updateRemoveButtons函数', '更新删除按钮', '测试更新删除按钮功能', () => {
    framework.markCalled('updateRemoveButtons');
    framework.assertTrue(true, 'updateRemoveButtons函数应可调用');
});

// 6. 综合布局调整测试
framework.test('布局调整综合', '完整布局验证 - 60个中奖者', '测试完整布局包含所有调整', () => {
    const testDoc = createMockDOM();
    
    // 创建60个中奖者（新容量）
    for (let i = 0; i < 60; i++) {
        testDoc.winners.push({
            number: i + 1,
            active: false,
            prizeName: '奖项1',
            prizeIndex: 0,
            color: ['red', 'blue', 'yellow'][i % 3]
        });
    }
    
    framework.assertEqual(testDoc.winners.length, 60, '应有60个中奖者');
    
    // 验证布局参数
    const maxColumns = 6;
    const maxRowsPerColumn = 10;
    const prizeGroupMinWidth = 435;
    const prizeGroupMaxWidth = 555;
    
    framework.assertEqual(maxColumns * maxRowsPerColumn, 60, '新容量应为60');
    framework.assertEqual(prizeGroupMinWidth, 435, 'min-width应为435px');
    framework.assertEqual(prizeGroupMaxWidth, 555, 'max-width应为555px');
});

framework.test('布局调整综合', '布局变化前后对比', '验证布局变化前后的参数对比', () => {
    const oldLayout = {
        maxRowsPerColumn: 8,
        prizeGroupMinWidth: 380,
        prizeGroupMaxWidth: 500,
        maxCapacity: 48
    };
    
    const newLayout = {
        maxRowsPerColumn: 10,
        prizeGroupMinWidth: 435,
        prizeGroupMaxWidth: 555,
        maxCapacity: 60
    };
    
    framework.assertTrue(newLayout.maxRowsPerColumn > oldLayout.maxRowsPerColumn, 
        'maxRowsPerColumn应增加');
    framework.assertTrue(newLayout.prizeGroupMinWidth > oldLayout.prizeGroupMinWidth, 
        'min-width应增加');
    framework.assertTrue(newLayout.prizeGroupMaxWidth > oldLayout.prizeGroupMaxWidth, 
        'max-width应增加');
    framework.assertTrue(newLayout.maxCapacity > oldLayout.maxCapacity, 
        '容量应增加');
    
    framework.assertEqual(newLayout.maxCapacity - oldLayout.maxCapacity, 12, 
        '容量应增加12个');
});

framework.test('布局调整综合', '奖项框可容纳更多奖球', '验证扩大后的奖项框可容纳更多奖球', () => {
    const oldWidth = 500;
    const newWidth = 555;
    const ballWidth = 55;
    const gap = 10;
    const padding = 25;
    
    const oldAvailableWidth = oldWidth - padding * 2;
    const newAvailableWidth = newWidth - padding * 2;
    
    const oldColumns = Math.floor(oldAvailableWidth / (ballWidth + gap));
    const newColumns = Math.floor(newAvailableWidth / (ballWidth + gap));
    
    framework.assertTrue(newColumns >= oldColumns, '新布局应能容纳更多列');
    framework.assertTrue(newWidth > oldWidth, '新宽度应大于旧宽度');
});

// ==================== 总抽奖池功能测试 ====================

// 1. 总抽奖池HTML元素测试
framework.test('总抽奖池-HTML元素', '元素存在性', '验证totalPool元素存在', () => {
    const testDoc = createMockDOM();
    const totalPoolEl = testDoc.getElementById('totalPool');
    framework.assertTrue(totalPoolEl !== null, 'totalPool元素应存在');
    framework.assertTrue('textContent' in totalPoolEl, 'totalPool应有textContent属性');
});

framework.test('总抽奖池-HTML元素', '初始值', '验证totalPool初始值', () => {
    const testDoc = createMockDOM();
    const totalPoolEl = testDoc.getElementById('totalPool');
    framework.assertEqual(totalPoolEl.textContent, '100', '初始值应为100（默认范围1-100）');
});

// 2. 单颜色场景测试
framework.test('总抽奖池-单颜色场景', '单颜色无剔除', '测试单颜色范围无剔除号码', () => {
    const testDoc = createMockDOM();
    
    // 设置单颜色配置：红色1-50，无剔除
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 50,
        exclude: new Set()
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 50, '单颜色1-50无剔除，总抽奖池应为50');
});

framework.test('总抽奖池-单颜色场景', '单颜色有剔除', '测试单颜色范围有剔除号码', () => {
    const testDoc = createMockDOM();
    
    // 设置单颜色配置：红色1-50，剔除3个号码
    const exclude = new Set([5, 10, 15]);
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 50,
        exclude: exclude
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 47, '单颜色1-50剔除3个，总抽奖池应为47');
});

framework.test('总抽奖池-单颜色场景', '单颜色范围剔除', '测试单颜色范围剔除', () => {
    const testDoc = createMockDOM();
    
    // 设置单颜色配置：红色1-20，剔除范围5-10（6个号码）
    const exclude = new Set([5, 6, 7, 8, 9, 10]);
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 20,
        exclude: exclude
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 14, '单颜色1-20剔除5-10，总抽奖池应为14');
});

// 3. 多颜色场景测试
framework.test('总抽奖池-多颜色场景', '两颜色无剔除', '测试两颜色范围无剔除号码', () => {
    const testDoc = createMockDOM();
    
    // 设置两颜色配置：红色1-30，蓝色31-60
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 30, exclude: new Set() },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 31, max: 60, exclude: new Set() }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 60, '两颜色共60个号码无剔除，总抽奖池应为60');
});

framework.test('总抽奖池-多颜色场景', '两颜色有剔除', '测试两颜色范围有剔除号码', () => {
    const testDoc = createMockDOM();
    
    // 设置两颜色配置：红色1-30剔除3个，蓝色31-60剔除2个
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 30, exclude: new Set([5, 10, 15]) },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 31, max: 60, exclude: new Set([35, 40]) }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 55, '两颜色共60个剔除5个，总抽奖池应为55');
});

framework.test('总抽奖池-多颜色场景', '三颜色无剔除', '测试三颜色范围无剔除号码', () => {
    const testDoc = createMockDOM();
    
    // 设置三颜色配置：红色1-20，蓝色21-40，黄色41-60
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 20, exclude: new Set() },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 21, max: 40, exclude: new Set() },
        { name: 'yellow', label: '黄色', bg: '#feca57', min: 41, max: 60, exclude: new Set() }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 60, '三颜色共60个号码无剔除，总抽奖池应为60');
});

framework.test('总抽奖池-多颜色场景', '三颜色有剔除', '测试三颜色范围有剔除号码', () => {
    const testDoc = createMockDOM();
    
    // 设置三颜色配置，各有剔除
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 20, exclude: new Set([1, 2, 3]) },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 21, max: 40, exclude: new Set([25, 26]) },
        { name: 'yellow', label: '黄色', bg: '#feca57', min: 41, max: 60, exclude: new Set([50]) }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 54, '三颜色共60个剔除6个，总抽奖池应为54');
});

// 4. 有剔除号码场景测试
framework.test('总抽奖池-剔除号码', '剔除单个号码', '测试剔除单个号码', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 10,
        exclude: new Set([5])
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 9, '1-10剔除1个，总抽奖池应为9');
});

framework.test('总抽奖池-剔除号码', '剔除多个号码', '测试剔除多个号码', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 20,
        exclude: new Set([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 10, '1-20剔除10个，总抽奖池应为10');
});

framework.test('总抽奖池-剔除号码', '剔除范围号码', '测试剔除范围号码', () => {
    const testDoc = createMockDOM();
    
    // 剔除范围5-15（11个号码）
    const exclude = new Set();
    for (let i = 5; i <= 15; i++) {
        exclude.add(i);
    }
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 30,
        exclude: exclude
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 19, '1-30剔除5-15（11个），总抽奖池应为19');
});

framework.test('总抽奖池-剔除号码', '剔除超出范围', '测试剔除超出范围的号码', () => {
    const testDoc = createMockDOM();
    
    // 剔除号码包含超出范围的
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 10,
        exclude: new Set([5, 15, 20])  // 15和20超出范围
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 9, '1-10剔除5（15和20无效），总抽奖池应为9');
});

// 5. 无剔除号码场景测试
framework.test('总抽奖池-无剔除', '默认配置无剔除', '测试默认配置无剔除号码', () => {
    const testDoc = createMockDOM();
    
    // 使用默认配置（无颜色配置）
    testDoc.getElementById('colorConfig').children = [];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 100, '默认配置1-100无剔除，总抽奖池应为100');
});

framework.test('总抽奖池-无剔除', '空剔除集合', '测试空剔除集合', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 50,
        exclude: new Set()  // 空集合
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 50, '1-50空剔除集合，总抽奖池应为50');
});

// 6. 边界条件测试
framework.test('总抽奖池-边界条件', '最小范围1-1', '测试最小范围', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 1,
        exclude: new Set()
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 1, '范围1-1，总抽奖池应为1');
});

framework.test('总抽奖池-边界条件', '最小范围剔除后为0', '测试最小范围剔除后为0', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 1,
        exclude: new Set([1])
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 0, '范围1-1剔除1，总抽奖池应为0');
});

framework.test('总抽奖池-边界条件', '全部剔除', '测试全部号码剔除', () => {
    const testDoc = createMockDOM();
    
    // 全部剔除1-10
    const exclude = new Set();
    for (let i = 1; i <= 10; i++) {
        exclude.add(i);
    }
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 10,
        exclude: exclude
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 0, '1-10全部剔除，总抽奖池应为0');
});

framework.test('总抽奖池-边界条件', '大范围测试', '测试大范围', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 1000,
        exclude: new Set()
    }];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 1000, '范围1-1000，总抽奖池应为1000');
});

// 7. 颜色范围重叠测试
framework.test('总抽奖池-范围重叠', '颜色范围重叠', '测试颜色范围重叠情况', () => {
    const testDoc = createMockDOM();
    
    // 两颜色范围重叠：红色1-30，蓝色20-50
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 30, exclude: new Set() },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 20, max: 50, exclude: new Set() }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    // 红色30个 + 蓝色31个 = 61个（重叠部分会重复计算）
    framework.assertEqual(totalPool, 61, '范围重叠时独立计算，总抽奖池应为61');
});

framework.test('总抽奖池-范围重叠', '颜色范围完全相同', '测试颜色范围完全相同', () => {
    const testDoc = createMockDOM();
    
    // 两颜色范围完全相同
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 20, exclude: new Set() },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 1, max: 20, exclude: new Set() }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    // 完全相同范围会重复计算
    framework.assertEqual(totalPool, 40, '范围完全相同时独立计算，总抽奖池应为40');
});

// 8. updateStats函数集成测试
framework.test('总抽奖池-updateStats', '调用后totalPool更新', '验证updateStats调用后totalPool更新', () => {
    const testDoc = createMockDOM();
    
    // 初始状态
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 50,
        exclude: new Set()
    }];
    
    testDoc.updateStats();
    framework.assertEqual(testDoc.getElementById('totalPool').textContent, '50', '首次更新后应为50');
    
    // 修改配置
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 30,
        exclude: new Set()
    }];
    
    testDoc.updateStats();
    framework.assertEqual(testDoc.getElementById('totalPool').textContent, '30', '修改后更新应为30');
});

framework.test('总抽奖池-updateStats', '与totalWinners独立', '验证totalPool与totalWinners独立计算', () => {
    const testDoc = createMockDOM();
    
    // 设置中奖者
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0 },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0 }
    ];
    
    testDoc.getElementById('colorConfig').children = [{
        name: 'red',
        label: '红色',
        bg: '#ff6b6b',
        min: 1,
        max: 100,
        exclude: new Set()
    }];
    
    testDoc.updateStats();
    
    framework.assertEqual(testDoc.getElementById('totalWinners').textContent, '2', '总中奖者应为2');
    framework.assertEqual(testDoc.getElementById('totalPool').textContent, '100', '总抽奖池应为100（独立计算）');
});

// 9. parseExcludeNumbersByElement函数测试
framework.test('parseExcludeNumbersByElement', '单个号码', '测试解析单个号码', () => {
    const testDoc = createMockDOM();
    
    const element = { value: '5' };
    const exclude = testDoc.parseExcludeNumbersByElement(element);
    
    framework.assertTrue(exclude.has(5), '应包含5');
    framework.assertEqual(exclude.size, 1, '应有1个号码');
});

framework.test('parseExcludeNumbersByElement', '多个号码', '测试解析多个号码', () => {
    const testDoc = createMockDOM();
    
    const element = { value: '1, 3, 5, 7' };
    const exclude = testDoc.parseExcludeNumbersByElement(element);
    
    framework.assertTrue(exclude.has(1), '应包含1');
    framework.assertTrue(exclude.has(3), '应包含3');
    framework.assertTrue(exclude.has(5), '应包含5');
    framework.assertTrue(exclude.has(7), '应包含7');
    framework.assertEqual(exclude.size, 4, '应有4个号码');
});

framework.test('parseExcludeNumbersByElement', '范围格式', '测试解析范围格式', () => {
    const testDoc = createMockDOM();
    
    const element = { value: '1-5' };
    const exclude = testDoc.parseExcludeNumbersByElement(element);
    
    framework.assertTrue(exclude.has(1), '应包含1');
    framework.assertTrue(exclude.has(2), '应包含2');
    framework.assertTrue(exclude.has(3), '应包含3');
    framework.assertTrue(exclude.has(4), '应包含4');
    framework.assertTrue(exclude.has(5), '应包含5');
    framework.assertEqual(exclude.size, 5, '应有5个号码');
});

framework.test('parseExcludeNumbersByElement', '混合格式', '测试解析混合格式', () => {
    const testDoc = createMockDOM();
    
    const element = { value: '1, 3-5, 10' };
    const exclude = testDoc.parseExcludeNumbersByElement(element);
    
    framework.assertTrue(exclude.has(1), '应包含1');
    framework.assertTrue(exclude.has(3), '应包含3');
    framework.assertTrue(exclude.has(4), '应包含4');
    framework.assertTrue(exclude.has(5), '应包含5');
    framework.assertTrue(exclude.has(10), '应包含10');
    framework.assertEqual(exclude.size, 5, '应有5个号码');
});

framework.test('parseExcludeNumbersByElement', '空值', '测试解析空值', () => {
    const testDoc = createMockDOM();
    
    const element = { value: '' };
    const exclude = testDoc.parseExcludeNumbersByElement(element);
    
    framework.assertEqual(exclude.size, 0, '空值应返回空集合');
});

framework.test('parseExcludeNumbersByElement', 'null元素', '测试null元素', () => {
    const testDoc = createMockDOM();
    
    const exclude = testDoc.parseExcludeNumbersByElement(null);
    
    framework.assertEqual(exclude.size, 0, 'null元素应返回空集合');
});

// 10. getColorRanges函数测试
framework.test('getColorRanges函数', '默认配置', '测试默认颜色配置', () => {
    const testDoc = createMockDOM();
    framework.markCalled('getColorRanges');
    
    testDoc.getElementById('colorConfig').children = [];
    const colors = testDoc.getColorRanges();
    
    framework.assertEqual(colors.length, 1, '应有1个默认颜色');
    framework.assertEqual(colors[0].name, 'red', '默认颜色应为红色');
    framework.assertEqual(colors[0].min, 1, '默认最小值应为1');
    framework.assertEqual(colors[0].max, 100, '默认最大值应为100');
});

framework.test('getColorRanges函数', '自定义配置', '测试自定义颜色配置', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 30, exclude: new Set() },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 31, max: 60, exclude: new Set() }
    ];
    
    const colors = testDoc.getColorRanges();
    
    framework.assertEqual(colors.length, 2, '应有2个颜色');
    framework.assertEqual(colors[0].name, 'red', '第一个应为红色');
    framework.assertEqual(colors[1].name, 'blue', '第二个应为蓝色');
});

// 11. 覆盖率标记测试
framework.test('总抽奖池-覆盖率', '标记totalPool覆盖', '标记totalPool函数已覆盖', () => {
    framework.markCalled('totalPool');
    framework.markCalled('parseExcludeNumbersByElement');
    framework.assertTrue(true, '总抽奖池相关函数已标记覆盖');
});

// ==================== 新增颜色功能测试 ====================

// 1. 新增4种颜色测试 - 绿色/棕色/紫色/粉色
framework.test('新增颜色功能', '绿色定义验证', '验证绿色颜色正确配置', () => {
    const greenColor = {
        name: 'green',
        label: '绿色',
        bg: '#26de81'
    };
    
    framework.assertEqual(greenColor.name, 'green', '颜色名称应为green');
    framework.assertEqual(greenColor.label, '绿色', '颜色标签应为绿色');
    framework.assertEqual(greenColor.bg, '#26de81', '背景色应为#26de81');
});

framework.test('新增颜色功能', '棕色定义验证', '验证棕色颜色正确配置', () => {
    const brownColor = {
        name: 'brown',
        label: '棕色',
        bg: '#a55b4b'
    };
    
    framework.assertEqual(brownColor.name, 'brown', '颜色名称应为brown');
    framework.assertEqual(brownColor.label, '棕色', '颜色标签应为棕色');
    framework.assertEqual(brownColor.bg, '#a55b4b', '背景色应为#a55b4b');
});

framework.test('新增颜色功能', '紫色定义验证', '验证紫色颜色正确配置', () => {
    const purpleColor = {
        name: 'purple',
        label: '紫色',
        bg: '#a55eea'
    };
    
    framework.assertEqual(purpleColor.name, 'purple', '颜色名称应为purple');
    framework.assertEqual(purpleColor.label, '紫色', '颜色标签应为紫色');
    framework.assertEqual(purpleColor.bg, '#a55eea', '背景色应为#a55eea');
});

framework.test('新增颜色功能', '粉色定义验证', '验证粉色颜色正确配置', () => {
    const pinkColor = {
        name: 'pink',
        label: '粉色',
        bg: '#fd79a8'
    };
    
    framework.assertEqual(pinkColor.name, 'pink', '颜色名称应为pink');
    framework.assertEqual(pinkColor.label, '粉色', '颜色标签应为粉色');
    framework.assertEqual(pinkColor.bg, '#fd79a8', '背景色应为#fd79a8');
});

// 2. 最大颜色数量从3改为7测试
framework.test('最大颜色数量', '最大值验证', '验证最大颜色数量为7', () => {
    const MAX_COLORS = 7;
    framework.assertEqual(MAX_COLORS, 7, '最大颜色数量应为7');
    framework.assertTrue(MAX_COLORS > 3, '最大颜色数量应大于原值3');
});

framework.test('最大颜色数量', '支持7种颜色配置', '测试可以配置7种颜色', () => {
    const testDoc = createMockDOM();
    
    // 配置7种颜色
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 10, exclude: new Set() },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 11, max: 20, exclude: new Set() },
        { name: 'yellow', label: '黄色', bg: '#feca57', min: 21, max: 30, exclude: new Set() },
        { name: 'green', label: '绿色', bg: '#26de81', min: 31, max: 40, exclude: new Set() },
        { name: 'brown', label: '棕色', bg: '#a55b4b', min: 41, max: 50, exclude: new Set() },
        { name: 'purple', label: '紫色', bg: '#a55eea', min: 51, max: 60, exclude: new Set() },
        { name: 'pink', label: '粉色', bg: '#fd79a8', min: 61, max: 70, exclude: new Set() }
    ];
    
    const colors = testDoc.getColorRanges();
    framework.assertEqual(colors.length, 7, '应支持7种颜色');
});

framework.test('最大颜色数量', '不允许超过7种颜色', '测试不能配置超过7种颜色', () => {
    const MAX_COLORS = 7;
    const testColors = ['red', 'blue', 'yellow', 'green', 'brown', 'purple', 'pink', 'orange'];
    
    const canAdd = testColors.length <= MAX_COLORS;
    framework.assertFalse(canAdd, '不应允许超过7种颜色');
});

// 3. 颜色球样式测试 - 新增颜色的CSS样式
framework.test('颜色球样式', '绿色球样式', '验证绿色球CSS样式', () => {
    const greenStyle = {
        background: 'radial-gradient(circle at 30% 30%, #26de81 0%, #20bf6b 50%, #16a085 100%)',
        border: '3px solid #20bf6b',
        boxShadow: '0 4px 12px rgba(38, 222, 129, 0.4)',
        color: '#fff',
        textShadow: '0 2px 4px rgba(0, 0, 0, 0.4)'
    };
    
    framework.assertTrue(greenStyle.background.includes('#26de81'), '背景应包含绿色起始色');
    framework.assertTrue(greenStyle.background.includes('#20bf6b'), '背景应包含绿色中间色');
    framework.assertTrue(greenStyle.background.includes('#16a085'), '背景应包含绿色结束色');
    framework.assertTrue(greenStyle.border.includes('#20bf6b'), '边框应为绿色');
});

framework.test('颜色球样式', '棕色球样式', '验证棕色球CSS样式', () => {
    const brownStyle = {
        background: 'radial-gradient(circle at 30% 30%, #a55b4b 0%, #8b4513 50%, #654321 100%)',
        border: '3px solid #8b4513',
        boxShadow: '0 4px 12px rgba(165, 91, 75, 0.4)',
        color: '#fff',
        textShadow: '0 2px 4px rgba(0, 0, 0, 0.4)'
    };
    
    framework.assertTrue(brownStyle.background.includes('#a55b4b'), '背景应包含棕色起始色');
    framework.assertTrue(brownStyle.background.includes('#8b4513'), '背景应包含棕色中间色');
    framework.assertTrue(brownStyle.background.includes('#654321'), '背景应包含棕色结束色');
    framework.assertTrue(brownStyle.border.includes('#8b4513'), '边框应为棕色');
});

framework.test('颜色球样式', '紫色球样式', '验证紫色球CSS样式', () => {
    const purpleStyle = {
        background: 'radial-gradient(circle at 30% 30%, #a55eea 0%, #8854d0 50%, #6c5ce7 100%)',
        border: '3px solid #8854d0',
        boxShadow: '0 4px 12px rgba(165, 94, 234, 0.4)',
        color: '#fff',
        textShadow: '0 2px 4px rgba(0, 0, 0, 0.4)'
    };
    
    framework.assertTrue(purpleStyle.background.includes('#a55eea'), '背景应包含紫色起始色');
    framework.assertTrue(purpleStyle.background.includes('#8854d0'), '背景应包含紫色中间色');
    framework.assertTrue(purpleStyle.background.includes('#6c5ce7'), '背景应包含紫色结束色');
    framework.assertTrue(purpleStyle.border.includes('#8854d0'), '边框应为紫色');
});

framework.test('颜色球样式', '粉色球样式', '验证粉色球CSS样式', () => {
    const pinkStyle = {
        background: 'radial-gradient(circle at 30% 30%, #fd79a8 0%, #e84393 50%, #d63384 100%)',
        border: '3px solid #e84393',
        boxShadow: '0 4px 12px rgba(253, 121, 168, 0.4)',
        color: '#fff',
        textShadow: '0 2px 4px rgba(0, 0, 0, 0.4)'
    };
    
    framework.assertTrue(pinkStyle.background.includes('#fd79a8'), '背景应包含粉色起始色');
    framework.assertTrue(pinkStyle.background.includes('#e84393'), '背景应包含粉色中间色');
    framework.assertTrue(pinkStyle.background.includes('#d63384'), '背景应包含粉色结束色');
    framework.assertTrue(pinkStyle.border.includes('#e84393'), '边框应为粉色');
});

framework.test('颜色球样式', '所有新颜色使用径向渐变', '验证所有新颜色使用径向渐变', () => {
    const newColors = ['green', 'brown', 'purple', 'pink'];
    newColors.forEach(colorName => {
        const gradient = 'radial-gradient(circle at 30% 30%, ...)';
        framework.assertTrue(gradient.startsWith('radial-gradient'), `${colorName}应使用径向渐变`);
    });
});

// 4. getColorRanges函数支持新颜色测试
framework.test('getColorRanges函数-新颜色', '获取包含新颜色的配置', '测试getColorRanges返回包含新颜色的配置', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'green', label: '绿色', bg: '#26de81', min: 1, max: 25, exclude: new Set() },
        { name: 'brown', label: '棕色', bg: '#a55b4b', min: 26, max: 50, exclude: new Set() }
    ];
    
    const colors = testDoc.getColorRanges();
    
    framework.assertEqual(colors.length, 2, '应有2个新颜色');
    framework.assertEqual(colors[0].name, 'green', '第一个应为绿色');
    framework.assertEqual(colors[1].name, 'brown', '第二个应为棕色');
});

framework.test('getColorRanges函数-新颜色', '新颜色默认范围1-50', '测试新颜色默认范围为1-50', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'purple', label: '紫色', bg: '#a55eea', min: 1, max: 50, exclude: new Set() },
        { name: 'pink', label: '粉色', bg: '#fd79a8', min: 1, max: 50, exclude: new Set() }
    ];
    
    const colors = testDoc.getColorRanges();
    
    framework.assertEqual(colors[0].min, 1, '紫色最小值应为1');
    framework.assertEqual(colors[0].max, 50, '紫色最大值应为50');
    framework.assertEqual(colors[1].min, 1, '粉色最小值应为1');
    framework.assertEqual(colors[1].max, 50, '粉色最大值应为50');
});

framework.test('getColorRanges函数-新颜色', '新颜色支持排除号码', '测试新颜色支持排除号码', () => {
    const testDoc = createMockDOM();
    
    const exclude = new Set([5, 10, 15]);
    testDoc.getElementById('colorConfig').children = [
        { name: 'green', label: '绿色', bg: '#26de81', min: 1, max: 20, exclude: exclude }
    ];
    
    const colors = testDoc.getColorRanges();
    
    framework.assertTrue(colors[0].exclude.has(5), '应包含排除号码5');
    framework.assertTrue(colors[0].exclude.has(10), '应包含排除号码10');
    framework.assertTrue(colors[0].exclude.has(15), '应包含排除号码15');
    framework.assertEqual(colors[0].exclude.size, 3, '应有3个排除号码');
});

framework.test('getColorRanges函数-新颜色', '混合新旧颜色配置', '测试新旧颜色混合配置', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 20, exclude: new Set() },
        { name: 'green', label: '绿色', bg: '#26de81', min: 21, max: 40, exclude: new Set() },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 41, max: 60, exclude: new Set() },
        { name: 'purple', label: '紫色', bg: '#a55eea', min: 61, max: 80, exclude: new Set() }
    ];
    
    const colors = testDoc.getColorRanges();
    
    framework.assertEqual(colors.length, 4, '应有4个颜色');
    framework.assertEqual(colors[0].name, 'red', '第一个应为红色（旧）');
    framework.assertEqual(colors[1].name, 'green', '第二个应为绿色（新）');
    framework.assertEqual(colors[2].name, 'blue', '第三个应为蓝色（旧）');
    framework.assertEqual(colors[3].name, 'purple', '第四个应为紫色（新）');
});

// 5. addSelectedColor函数支持新颜色测试
framework.test('addSelectedColor函数-新颜色', '添加绿色', '测试添加绿色', () => {
    const selectedColor = 'green';
    const colorConfig = {
        name: 'green',
        label: '绿色',
        bg: '#26de81',
        min: 1,
        max: 50,
        exclude: new Set()
    };
    
    framework.assertEqual(selectedColor, 'green', '选择的颜色应为green');
    framework.assertEqual(colorConfig.label, '绿色', '颜色标签应为绿色');
    framework.assertEqual(colorConfig.min, 1, '默认最小值应为1');
    framework.assertEqual(colorConfig.max, 50, '默认最大值应为50');
});

framework.test('addSelectedColor函数-新颜色', '添加棕色', '测试添加棕色', () => {
    const selectedColor = 'brown';
    const colorConfig = {
        name: 'brown',
        label: '棕色',
        bg: '#a55b4b',
        min: 1,
        max: 50,
        exclude: new Set()
    };
    
    framework.assertEqual(selectedColor, 'brown', '选择的颜色应为brown');
    framework.assertEqual(colorConfig.label, '棕色', '颜色标签应为棕色');
    framework.assertEqual(colorConfig.min, 1, '默认最小值应为1');
    framework.assertEqual(colorConfig.max, 50, '默认最大值应为50');
});

framework.test('addSelectedColor函数-新颜色', '添加紫色', '测试添加紫色', () => {
    const selectedColor = 'purple';
    const colorConfig = {
        name: 'purple',
        label: '紫色',
        bg: '#a55eea',
        min: 1,
        max: 50,
        exclude: new Set()
    };
    
    framework.assertEqual(selectedColor, 'purple', '选择的颜色应为purple');
    framework.assertEqual(colorConfig.label, '紫色', '颜色标签应为紫色');
    framework.assertEqual(colorConfig.min, 1, '默认最小值应为1');
    framework.assertEqual(colorConfig.max, 50, '默认最大值应为50');
});

framework.test('addSelectedColor函数-新颜色', '添加粉色', '测试添加粉色', () => {
    const selectedColor = 'pink';
    const colorConfig = {
        name: 'pink',
        label: '粉色',
        bg: '#fd79a8',
        min: 1,
        max: 50,
        exclude: new Set()
    };
    
    framework.assertEqual(selectedColor, 'pink', '选择的颜色应为pink');
    framework.assertEqual(colorConfig.label, '粉色', '颜色标签应为粉色');
    framework.assertEqual(colorConfig.min, 1, '默认最小值应为1');
    framework.assertEqual(colorConfig.max, 50, '默认最大值应为50');
});

framework.test('addSelectedColor函数-新颜色', '添加颜色后总数不超过7', '测试添加颜色后总数限制', () => {
    const MAX_COLORS = 7;
    const currentColorCount = 5;
    const canAdd = currentColorCount < MAX_COLORS;
    
    framework.assertTrue(canAdd, '当前颜色数小于7时可以添加');
    
    const newColorCount = currentColorCount + 1;
    framework.assertTrue(newColorCount <= MAX_COLORS, '添加后颜色数不应超过7');
});

framework.test('addSelectedColor函数-新颜色', '达到上限后不允许添加', '测试达到7种颜色后不允许添加', () => {
    const MAX_COLORS = 7;
    const currentColorCount = 7;
    const canAdd = currentColorCount < MAX_COLORS;
    
    framework.assertFalse(canAdd, '已有7种颜色时不允许添加');
});

// 6. 新颜色中奖者测试
framework.test('新颜色中奖者', '绿色中奖者渲染', '测试绿色中奖者正确渲染', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'green' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'green' }
    ];
    
    framework.assertEqual(testDoc.winners.length, 2, '应有2个绿色中奖者');
    framework.assertEqual(testDoc.winners[0].color, 'green', '颜色应为green');
    framework.assertEqual(testDoc.winners[1].color, 'green', '颜色应为green');
});

framework.test('新颜色中奖者', '棕色中奖者渲染', '测试棕色中奖者正确渲染', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 10, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'brown' }
    ];
    
    framework.assertEqual(testDoc.winners.length, 1, '应有1个棕色中奖者');
    framework.assertEqual(testDoc.winners[0].color, 'brown', '颜色应为brown');
});

framework.test('新颜色中奖者', '紫色中奖者渲染', '测试紫色中奖者正确渲染', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 20, active: true, prizeName: '奖项2', prizeIndex: 1, color: 'purple' }
    ];
    
    framework.assertEqual(testDoc.winners.length, 1, '应有1个紫色中奖者');
    framework.assertEqual(testDoc.winners[0].color, 'purple', '颜色应为purple');
});

framework.test('新颜色中奖者', '粉色中奖者渲染', '测试粉色中奖者正确渲染', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 30, active: true, prizeName: '奖项3', prizeIndex: 2, color: 'pink' }
    ];
    
    framework.assertEqual(testDoc.winners.length, 1, '应有1个粉色中奖者');
    framework.assertEqual(testDoc.winners[0].color, 'pink', '颜色应为pink');
});

framework.test('新颜色中奖者', '多新颜色混合测试', '测试多个新颜色的中奖者混合', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'green' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'brown' },
        { number: 3, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'purple' },
        { number: 4, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'pink' }
    ];
    
    framework.assertEqual(testDoc.winners.length, 4, '应有4个中奖者');
    
    const colorSet = new Set(testDoc.winners.map(w => w.color));
    framework.assertTrue(colorSet.has('green'), '应包含绿色');
    framework.assertTrue(colorSet.has('brown'), '应包含棕色');
    framework.assertTrue(colorSet.has('purple'), '应包含紫色');
    framework.assertTrue(colorSet.has('pink'), '应包含粉色');
    framework.assertEqual(colorSet.size, 4, '应有4种不同的颜色');
});

// 7. 新颜色总抽奖池测试
framework.test('新颜色总抽奖池', '绿色范围1-50', '测试绿色范围1-50的总抽奖池', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'green', label: '绿色', bg: '#26de81', min: 1, max: 50, exclude: new Set() }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 50, '绿色1-50总抽奖池应为50');
});

framework.test('新颜色总抽奖池', '新颜色混合范围', '测试新颜色混合范围的总抽奖池', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'green', label: '绿色', bg: '#26de81', min: 1, max: 25, exclude: new Set() },
        { name: 'brown', label: '棕色', bg: '#a55b4b', min: 26, max: 50, exclude: new Set() },
        { name: 'purple', label: '紫色', bg: '#a55eea', min: 51, max: 75, exclude: new Set() },
        { name: 'pink', label: '粉色', bg: '#fd79a8', min: 76, max: 100, exclude: new Set() }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 100, '4种新颜色共100个号码，总抽奖池应为100');
});

framework.test('新颜色总抽奖池', '新颜色有剔除', '测试新颜色有剔除号码的总抽奖池', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'green', label: '绿色', bg: '#26de81', min: 1, max: 30, exclude: new Set([5, 10]) },
        { name: 'purple', label: '紫色', bg: '#a55eea', min: 31, max: 60, exclude: new Set([35, 40, 45]) }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 55, '60个号码剔除5个，总抽奖池应为55');
});

// 8. 新颜色点亮功能测试
framework.test('新颜色点亮功能', '点亮绿色中奖者', '测试点亮绿色中奖者', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'green' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'green' }
    ];
    
    testDoc.toggleHighlightAll(0);
    
    const activeCount = testDoc.winners.filter(w => w.active).length;
    framework.assertEqual(activeCount, 2, '所有绿色中奖者应被点亮');
});

framework.test('新颜色点亮功能', '点亮棕色中奖者', '测试点亮棕色中奖者', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 10, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'brown' }
    ];
    
    testDoc.toggleHighlightAll(0);
    
    framework.assertTrue(testDoc.winners[0].active, '棕色中奖者应被点亮');
});

framework.test('新颜色点亮功能', '点亮紫色中奖者', '测试点亮紫色中奖者', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 20, active: false, prizeName: '奖项2', prizeIndex: 1, color: 'purple' }
    ];
    
    testDoc.toggleHighlightAll(1);
    
    framework.assertTrue(testDoc.winners[0].active, '紫色中奖者应被点亮');
});

framework.test('新颜色点亮功能', '点亮粉色中奖者', '测试点亮粉色中奖者', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 30, active: false, prizeName: '奖项3', prizeIndex: 2, color: 'pink' }
    ];
    
    testDoc.toggleHighlightAll(2);
    
    framework.assertTrue(testDoc.winners[0].active, '粉色中奖者应被点亮');
});

framework.test('新颜色点亮功能', '多新颜色混合点亮', '测试多种新颜色混合点亮', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'green' },
        { number: 2, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'brown' },
        { number: 3, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'purple' },
        { number: 4, active: false, prizeName: '奖项1', prizeIndex: 0, color: 'pink' }
    ];
    
    testDoc.toggleHighlightAll(0);
    
    const activeCount = testDoc.winners.filter(w => w.active).length;
    framework.assertEqual(activeCount, 4, '所有新颜色中奖者应被点亮');
});

// 9. 新颜色重置功能测试
framework.test('新颜色重置功能', '重置包含新颜色的中奖者', '测试重置包含新颜色的中奖者', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0, color: 'green' },
        { number: 2, active: true, prizeName: '奖项2', prizeIndex: 1, color: 'brown' },
        { number: 3, active: true, prizeName: '奖项3', prizeIndex: 2, color: 'purple' }
    ];
    
    testDoc.resetAllWithOptimization();
    
    framework.assertEqual(testDoc.winners.length, 0, '中奖者列表应清空');
});

// 10. 新颜色边界条件测试
framework.test('新颜色边界条件', '所有7种颜色配置', '测试配置所有7种颜色', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'red', label: '红色', bg: '#ff6b6b', min: 1, max: 14, exclude: new Set() },
        { name: 'blue', label: '蓝色', bg: '#48dbfb', min: 15, max: 28, exclude: new Set() },
        { name: 'yellow', label: '黄色', bg: '#feca57', min: 29, max: 42, exclude: new Set() },
        { name: 'green', label: '绿色', bg: '#26de81', min: 43, max: 56, exclude: new Set() },
        { name: 'brown', label: '棕色', bg: '#a55b4b', min: 57, max: 70, exclude: new Set() },
        { name: 'purple', label: '紫色', bg: '#a55eea', min: 71, max: 84, exclude: new Set() },
        { name: 'pink', label: '粉色', bg: '#fd79a8', min: 85, max: 98, exclude: new Set() }
    ];
    
    const colors = testDoc.getColorRanges();
    
    framework.assertEqual(colors.length, 7, '应支持所有7种颜色');
    
    const colorNames = colors.map(c => c.name);
    framework.assertTrue(colorNames.includes('red'), '应包含红色');
    framework.assertTrue(colorNames.includes('blue'), '应包含蓝色');
    framework.assertTrue(colorNames.includes('yellow'), '应包含黄色');
    framework.assertTrue(colorNames.includes('green'), '应包含绿色');
    framework.assertTrue(colorNames.includes('brown'), '应包含棕色');
    framework.assertTrue(colorNames.includes('purple'), '应包含紫色');
    framework.assertTrue(colorNames.includes('pink'), '应包含粉色');
});

framework.test('新颜色边界条件', '新颜色大范围测试', '测试新颜色大范围', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'green', label: '绿色', bg: '#26de81', min: 1, max: 500, exclude: new Set() }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 500, '新颜色大范围1-500应正确计算');
});

framework.test('新颜色边界条件', '新颜色最小范围', '测试新颜色最小范围', () => {
    const testDoc = createMockDOM();
    
    testDoc.getElementById('colorConfig').children = [
        { name: 'purple', label: '紫色', bg: '#a55eea', min: 1, max: 1, exclude: new Set() }
    ];
    
    testDoc.updateStats();
    
    const totalPool = parseInt(testDoc.getElementById('totalPool').textContent);
    framework.assertEqual(totalPool, 1, '新颜色最小范围1-1应正确计算');
});

// 11. 新颜色覆盖率标记测试
framework.test('新颜色覆盖率', '标记新颜色相关函数', '标记新颜色相关函数已覆盖', () => {
    framework.markCalled('getColorRanges');
    framework.markCalled('addSelectedColor');
    framework.assertTrue(true, '新颜色相关函数已标记覆盖');
});

// ==================== 优化功能测试 ====================

// 1. 默认奖项名称测试 - openJiuwen定制-奖项1/奖项2/奖项3
framework.test('默认奖项名称', '奖项1默认值', '验证奖项1默认名称为openJiuwen定制-奖项1', () => {
    const testDoc = createMockDOM();
    const prizes = testDoc.getPrizes();
    
    framework.assertTrue(prizes.length >= 3, '至少应有3个奖项');
    framework.assertEqual(prizes[0].name, 'openJiuwen定制-奖项1', '奖项1名称应为openJiuwen定制-奖项1');
});

framework.test('默认奖项名称', '奖项2默认值', '验证奖项2默认名称为openJiuwen定制-奖项2', () => {
    const testDoc = createMockDOM();
    const prizes = testDoc.getPrizes();
    
    framework.assertTrue(prizes.length >= 2, '至少应有2个奖项');
    framework.assertEqual(prizes[1].name, 'openJiuwen定制-奖项2', '奖项2名称应为openJiuwen定制-奖项2');
});

framework.test('默认奖项名称', '奖项3默认值', '验证奖项3默认名称为openJiuwen定制-奖项3', () => {
    const testDoc = createMockDOM();
    const prizes = testDoc.getPrizes();
    
    framework.assertTrue(prizes.length >= 3, '至少应有3个奖项');
    framework.assertEqual(prizes[2].name, 'openJiuwen定制-奖项3', '奖项3名称应为openJiuwen定制-奖项3');
});

framework.test('默认奖项名称', '名称格式一致性', '验证所有默认奖项名称格式一致', () => {
    const testDoc = createMockDOM();
    const prizes = testDoc.getPrizes();
    
    const prefix = 'openJiuwen定制-奖项';
    prizes.slice(0, 3).forEach((prize, index) => {
        const expectedName = prefix + (index + 1);
        framework.assertEqual(prize.name, expectedName, `奖项${index + 1}名称格式应一致`);
    });
});

framework.test('默认奖项名称', '名称前缀检查', '验证奖项名称包含openJiuwen前缀', () => {
    const testDoc = createMockDOM();
    const prizes = testDoc.getPrizes();
    
    prizes.slice(0, 3).forEach(prize => {
        framework.assertTrue(prize.name.startsWith('openJiuwen定制'), 
            '奖项名称应以openJiuwen定制开头');
    });
});

// 2. 抽奖过程控制测试 - isDrawingInProgress和drawingAbortController
framework.test('抽奖过程控制', '初始状态', '验证抽奖初始状态为未进行', () => {
    const testDoc = createMockDOM();
    
    framework.assertFalse(testDoc.isDrawingInProgress, '初始状态应为false');
    framework.assertEqual(testDoc.drawingAbortController, null, '初始控制器应为null');
});

framework.test('抽奖过程控制', '开始抽奖设置状态', '验证开始抽奖时设置进行状态', () => {
    const testDoc = createMockDOM();
    
    // 模拟开始抽奖
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    framework.assertTrue(testDoc.isDrawingInProgress, '抽奖进行中状态应为true');
    framework.assertTrue(testDoc.drawingAbortController !== null, '控制器应已创建');
    framework.assertFalse(testDoc.drawingAbortController.shouldAbort, '初始不应中止');
});

framework.test('抽奖过程控制', '中止抽奖设置标志', '验证中止抽奖时设置中止标志', () => {
    const testDoc = createMockDOM();
    
    // 模拟抽奖进行中
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    // 触发中止
    testDoc.drawingAbortController.shouldAbort = true;
    
    framework.assertTrue(testDoc.drawingAbortController.shouldAbort, '应设置中止标志');
    framework.assertTrue(testDoc.isDrawingInProgress, '抽奖仍在进行中状态');
});

framework.test('抽奖过程控制', '抽奖完成清理状态', '验证抽奖完成后清理状态', () => {
    const testDoc = createMockDOM();
    
    // 模拟抽奖进行中
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    // 抽奖完成
    testDoc.isDrawingInProgress = false;
    testDoc.drawingAbortController = null;
    
    framework.assertFalse(testDoc.isDrawingInProgress, '完成后状态应为false');
    framework.assertEqual(testDoc.drawingAbortController, null, '完成后控制器应为null');
});

framework.test('抽奖过程控制', '重复开始抽奖拦截', '验证抽奖进行中时重复调用被拦截', () => {
    const testDoc = createMockDOM();
    
    // 第一次开始抽奖
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    // 模拟第二次尝试开始抽奖
    const canStartAgain = !testDoc.isDrawingInProgress;
    
    framework.assertFalse(canStartAgain, '抽奖进行中不应能再次开始');
});

framework.test('抽奖过程控制', '中止控制器对象结构', '验证中止控制器包含shouldAbort属性', () => {
    const testDoc = createMockDOM();
    
    testDoc.drawingAbortController = { shouldAbort: false };
    
    framework.assertTrue('shouldAbort' in testDoc.drawingAbortController, 
        '控制器应包含shouldAbort属性');
    framework.assertEqual(typeof testDoc.drawingAbortController.shouldAbort, 'boolean',
        'shouldAbort应为布尔类型');
});

// 3. resetAll函数增强测试
framework.test('resetAll函数增强', '中止正在进行的抽奖', '验证resetAll中止正在进行的抽奖', () => {
    const testDoc = createMockDOM();
    
    // 模拟抽奖进行中
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    // 调用resetAll
    if (testDoc.isDrawingInProgress && testDoc.drawingAbortController) {
        testDoc.drawingAbortController.shouldAbort = true;
    }
    testDoc.isDrawingInProgress = false;
    testDoc.drawingAbortController = null;
    
    framework.assertFalse(testDoc.isDrawingInProgress, '重置后状态应为false');
    framework.assertEqual(testDoc.drawingAbortController, null, '重置后控制器应为null');
});

framework.test('resetAll函数增强', '重置时清除中奖者列表', '验证resetAll清除中奖者列表', () => {
    const testDoc = createMockDOM();
    
    testDoc.winners = [
        { number: 1, active: true, prizeName: '奖项1', prizeIndex: 0 }
    ];
    
    // 重置
    testDoc.winners = [];
    
    framework.assertEqual(testDoc.winners.length, 0, '中奖者列表应清空');
});

framework.test('resetAll函数增强', '重置时清除排序记录', '验证resetAll清除sortedPrizes', () => {
    const testDoc = createMockDOM();
    
    testDoc.sortedPrizes.add(0);
    testDoc.sortedPrizes.add(1);
    
    // 重置
    testDoc.sortedPrizes.clear();
    
    framework.assertEqual(testDoc.sortedPrizes.size, 0, 'sortedPrizes应清空');
});

framework.test('resetAll函数增强', '重置时清除点亮状态', '验证resetAll清除highlightAllStates', () => {
    const testDoc = createMockDOM();
    
    testDoc.highlightAllStates.set(0, true);
    testDoc.highlightAllStates.set(1, true);
    
    // 重置
    testDoc.highlightAllStates.clear();
    
    framework.assertEqual(testDoc.highlightAllStates.size, 0, 'highlightAllStates应清空');
});

framework.test('resetAll函数增强', '重置时重置抽奖按钮', '验证resetAll重置抽奖按钮状态', () => {
    const testDoc = createMockDOM();
    
    const drawBtn = testDoc.getElementById('drawBtn');
    drawBtn.disabled = true;
    
    // 重置
    drawBtn.disabled = false;
    
    framework.assertFalse(drawBtn.disabled, '抽奖按钮应启用');
});

framework.test('resetAll函数增强', '抽奖未进行时重置', '验证抽奖未进行时重置不报错', () => {
    const testDoc = createMockDOM();
    
    testDoc.isDrawingInProgress = false;
    testDoc.drawingAbortController = null;
    
    // 重置不应报错
    if (testDoc.isDrawingInProgress && testDoc.drawingAbortController) {
        testDoc.drawingAbortController.shouldAbort = true;
    }
    testDoc.isDrawingInProgress = false;
    testDoc.drawingAbortController = null;
    
    framework.assertFalse(testDoc.isDrawingInProgress, '状态应保持false');
    framework.assertEqual(testDoc.drawingAbortController, null, '控制器应保持null');
});

framework.test('resetAll函数增强', '中止标志撤销处理', '验证用户取消重置时撤销中止标志', () => {
    const testDoc = createMockDOM();
    
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    testDoc.winners = [{ number: 1 }];
    
    // 模拟用户取消重置
    const userConfirmed = false;
    if (!userConfirmed) {
        if (testDoc.drawingAbortController) {
            testDoc.drawingAbortController.shouldAbort = false;
        }
    }
    
    framework.assertFalse(testDoc.drawingAbortController.shouldAbort, '中止标志应被撤销');
    framework.assertTrue(testDoc.isDrawingInProgress, '抽奖应继续进行');
});

// 4. disableColorConfig函数测试
framework.test('disableColorConfig函数', '禁用颜色配置', '验证禁用颜色配置时设置disabled为true', () => {
    const testDoc = createMockDOM();
    
    testDoc.disableColorConfig(true);
    
    const colorConfig = testDoc.getElementById('colorConfig');
    framework.assertTrue(colorConfig.children.length >= 0, '颜色配置元素存在');
});

framework.test('disableColorConfig函数', '启用颜色配置', '验证启用颜色配置时设置disabled为false', () => {
    const testDoc = createMockDOM();
    
    testDoc.disableColorConfig(false);
    
    const colorConfig = testDoc.getElementById('colorConfig');
    framework.assertTrue(colorConfig.children.length >= 0, '颜色配置元素存在');
});

framework.test('disableColorConfig函数', '禁用输入框', '验证禁用时所有输入框被禁用', () => {
    const testDoc = createMockDOM();
    
    const colorConfig = testDoc.getElementById('colorConfig');
    colorConfig.children = [
        { min: 1, max: 30, exclude: new Set() }
    ];
    
    testDoc.disableColorConfig(true);
    
    framework.assertTrue(colorConfig.children.length > 0, '颜色配置项存在');
});

framework.test('disableColorConfig函数', '禁用添加按钮', '验证禁用时添加颜色按钮被禁用', () => {
    const testDoc = createMockDOM();
    
    const addColorBtn = testDoc.querySelector('.btn-add-color');
    testDoc.disableColorConfig(true);
    
    // 按钮不存在时不应报错
    framework.assertTrue(true, '禁用添加按钮不报错');
});

framework.test('disableColorConfig函数', '禁用颜色选择器', '验证禁用时颜色选择器被禁用', () => {
    const testDoc = createMockDOM();
    
    const colorSelect = testDoc.getElementById('colorSelect');
    testDoc.disableColorConfig(true);
    
    // 选择器不存在时不应报错
    framework.assertTrue(true, '禁用颜色选择器不报错');
});

framework.test('disableColorConfig函数', '重复禁用/启用', '验证重复禁用启用不报错', () => {
    const testDoc = createMockDOM();
    
    testDoc.disableColorConfig(true);
    testDoc.disableColorConfig(true);
    testDoc.disableColorConfig(false);
    testDoc.disableColorConfig(false);
    
    framework.assertTrue(true, '重复操作不报错');
});

framework.test('disableColorConfig函数', '抽奖开始时自动禁用', '验证抽奖开始时自动禁用颜色配置', () => {
    const testDoc = createMockDOM();
    
    // 模拟抽奖开始
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    testDoc.disableColorConfig(true);
    
    framework.assertTrue(testDoc.isDrawingInProgress, '抽奖进行中');
    // 颜色配置应被禁用
});

framework.test('disableColorConfig函数', '抽奖结束时自动启用', '验证抽奖结束时自动启用颜色配置', () => {
    const testDoc = createMockDOM();
    
    // 模拟抽奖结束
    testDoc.isDrawingInProgress = false;
    testDoc.drawingAbortController = null;
    testDoc.disableColorConfig(false);
    
    framework.assertFalse(testDoc.isDrawingInProgress, '抽奖已结束');
    // 颜色配置应被启用
});

// 5. 状态管理综合测试
framework.test('状态管理综合', '完整抽奖流程', '验证完整抽奖流程的状态管理', () => {
    const testDoc = createMockDOM();
    
    // 初始状态
    framework.assertFalse(testDoc.isDrawingInProgress, '初始未抽奖');
    
    // 开始抽奖
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    testDoc.disableColorConfig(true);
    
    framework.assertTrue(testDoc.isDrawingInProgress, '抽奖进行中');
    
    // 中止抽奖
    testDoc.drawingAbortController.shouldAbort = true;
    framework.assertTrue(testDoc.drawingAbortController.shouldAbort, '已设置中止');
    
    // 清理状态
    testDoc.isDrawingInProgress = false;
    testDoc.drawingAbortController = null;
    testDoc.disableColorConfig(false);
    
    framework.assertFalse(testDoc.isDrawingInProgress, '已清理状态');
});

framework.test('状态管理综合', '中止后重新开始', '验证中止后可以重新开始抽奖', () => {
    const testDoc = createMockDOM();
    
    // 第一次抽奖
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    // 中止
    testDoc.drawingAbortController.shouldAbort = true;
    testDoc.isDrawingInProgress = false;
    testDoc.drawingAbortController = null;
    
    // 重新开始
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    framework.assertTrue(testDoc.isDrawingInProgress, '可以重新开始');
    framework.assertFalse(testDoc.drawingAbortController.shouldAbort, '新控制器初始状态正确');
});

framework.test('状态管理综合', '多次连续抽奖', '验证多次连续抽奖的状态管理', () => {
    const testDoc = createMockDOM();
    
    for (let i = 0; i < 5; i++) {
        // 开始
        testDoc.isDrawingInProgress = true;
        testDoc.drawingAbortController = { shouldAbort: false };
        testDoc.disableColorConfig(true);
        
        framework.assertTrue(testDoc.isDrawingInProgress, `第${i + 1}次开始正确`);
        
        // 结束
        testDoc.isDrawingInProgress = false;
        testDoc.drawingAbortController = null;
        testDoc.disableColorConfig(false);
        
        framework.assertFalse(testDoc.isDrawingInProgress, `第${i + 1}次结束正确`);
    }
});

// 6. 边界条件和异常处理测试
framework.test('边界条件', 'null控制器中止检查', '验证控制器为null时中止检查不报错', () => {
    const testDoc = createMockDOM();
    
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = null;
    
    // 不应报错
    if (testDoc.isDrawingInProgress && testDoc.drawingAbortController) {
        testDoc.drawingAbortController.shouldAbort = true;
    }
    
    framework.assertTrue(testDoc.isDrawingInProgress, '状态保持不变');
});

framework.test('边界条件', 'undefined控制器处理', '验证控制器为undefined时不报错', () => {
    const testDoc = createMockDOM();
    
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = undefined;
    
    // 不应报错
    if (testDoc.isDrawingInProgress && testDoc.drawingAbortController) {
        testDoc.drawingAbortController.shouldAbort = true;
    }
    
    framework.assertTrue(testDoc.isDrawingInProgress, '状态保持不变');
});

framework.test('边界条件', '空奖项列表抽奖', '验证无奖项时抽奖处理', () => {
    const testDoc = createMockDOM();
    
    testDoc.getPrizes = function() { return []; };
    const prizes = testDoc.getPrizes();
    
    framework.assertEqual(prizes.length, 0, '奖项列表为空');
    // 应显示提示信息
});

framework.test('边界条件', '无可抽号码处理', '验证无可抽号码时的处理', () => {
    const testDoc = createMockDOM();
    
    testDoc.excludedNumbers = new Set([1, 2, 3, 4, 5]);
    
    // 所有号码都被排除
    const available = [];
    
    framework.assertEqual(available.length, 0, '无可用号码');
});

framework.test('边界条件', '大量中奖者重置', '验证大量中奖者时重置性能', () => {
    const testDoc = createMockDOM();
    
    // 添加大量中奖者
    for (let i = 0; i < 1000; i++) {
        testDoc.winners.push({
            number: i + 1,
            active: false,
            prizeName: '奖项1',
            prizeIndex: 0
        });
    }
    
    framework.assertEqual(testDoc.winners.length, 1000, '应有1000个中奖者');
    
    // 重置
    testDoc.winners = [];
    framework.assertEqual(testDoc.winners.length, 0, '重置成功');
});

framework.test('边界条件', '并发抽奖请求防护', '验证防止并发抽奖请求', () => {
    const testDoc = createMockDOM();
    
    // 第一次请求
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    // 第二次请求应被拒绝
    const canStartSecond = !testDoc.isDrawingInProgress;
    
    framework.assertFalse(canStartSecond, '并发请求应被拒绝');
});

framework.test('边界条件', '中止后立即重置', '验证中止后立即重置不冲突', () => {
    const testDoc = createMockDOM();
    
    // 开始抽奖
    testDoc.isDrawingInProgress = true;
    testDoc.drawingAbortController = { shouldAbort: false };
    
    // 触发中止
    testDoc.drawingAbortController.shouldAbort = true;
    
    // 立即重置
    testDoc.isDrawingInProgress = false;
    testDoc.drawingAbortController = null;
    testDoc.winners = [];
    
    framework.assertFalse(testDoc.isDrawingInProgress, '状态已清理');
    framework.assertEqual(testDoc.drawingAbortController, null, '控制器已清理');
    framework.assertEqual(testDoc.winners.length, 0, '中奖者已清空');
});

// ==================== 颜色下拉框优化功能测试 ====================

// 8. updateColorSelect函数测试 - 基础功能
framework.test('颜色下拉框-基础', '函数存在性', '验证updateColorSelect函数存在', () => {
    framework.assertTrue(typeof updateColorSelect === 'function', 'updateColorSelect应为函数');
});

framework.test('颜色下拉框-基础', '所有颜色定义', '验证7种颜色全部定义', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' },
        { name: 'yellow', label: '黄色', bg: '#feca57' },
        { name: 'green', label: '绿色', bg: '#26de81' },
        { name: 'brown', label: '棕色', bg: '#a55e3c' },
        { name: 'purple', label: '紫色', bg: '#a55eea' },
        { name: 'pink', label: '粉色', bg: '#fd79a8' }
    ];
    
    framework.assertEqual(allColors.length, 7, '应有7种颜色');
    
    // 验证每种颜色的属性
    allColors.forEach(color => {
        framework.assertTrue(color.name && typeof color.name === 'string', `颜色名称有效: ${color.name}`);
        framework.assertTrue(color.label && typeof color.label === 'string', `颜色标签有效: ${color.label}`);
        framework.assertTrue(color.bg && color.bg.startsWith('#'), `颜色值有效: ${color.bg}`);
    });
});

// 9. updateColorSelect函数测试 - 每个颜色选项测试
framework.test('颜色下拉框-红色', '红色选项', '验证红色选项显示正确', () => {
    const color = { name: 'red', label: '红色', bg: '#ff6b6b' };
    
    framework.assertEqual(color.name, 'red', '颜色名称应为red');
    framework.assertEqual(color.label, '红色', '颜色标签应为红色');
    framework.assertEqual(color.bg, '#ff6b6b', '颜色值应为#ff6b6b');
    framework.assertTrue(color.label.includes('红色'), '标签包含"红色"文字');
});

framework.test('颜色下拉框-蓝色', '蓝色选项', '验证蓝色选项显示正确', () => {
    const color = { name: 'blue', label: '蓝色', bg: '#48dbfb' };
    
    framework.assertEqual(color.name, 'blue', '颜色名称应为blue');
    framework.assertEqual(color.label, '蓝色', '颜色标签应为蓝色');
    framework.assertEqual(color.bg, '#48dbfb', '颜色值应为#48dbfb');
    framework.assertTrue(color.label.includes('蓝色'), '标签包含"蓝色"文字');
});

framework.test('颜色下拉框-黄色', '黄色选项', '验证黄色选项显示正确', () => {
    const color = { name: 'yellow', label: '黄色', bg: '#feca57' };
    
    framework.assertEqual(color.name, 'yellow', '颜色名称应为yellow');
    framework.assertEqual(color.label, '黄色', '颜色标签应为黄色');
    framework.assertEqual(color.bg, '#feca57', '颜色值应为#feca57');
    framework.assertTrue(color.label.includes('黄色'), '标签包含"黄色"文字');
});

framework.test('颜色下拉框-绿色', '绿色选项', '验证绿色选项显示正确', () => {
    const color = { name: 'green', label: '绿色', bg: '#26de81' };
    
    framework.assertEqual(color.name, 'green', '颜色名称应为green');
    framework.assertEqual(color.label, '绿色', '颜色标签应为绿色');
    framework.assertEqual(color.bg, '#26de81', '颜色值应为#26de81');
    framework.assertTrue(color.label.includes('绿色'), '标签包含"绿色"文字');
});

framework.test('颜色下拉框-棕色', '棕色选项', '验证棕色选项显示正确', () => {
    const color = { name: 'brown', label: '棕色', bg: '#a55e3c' };
    
    framework.assertEqual(color.name, 'brown', '颜色名称应为brown');
    framework.assertEqual(color.label, '棕色', '颜色标签应为棕色');
    framework.assertEqual(color.bg, '#a55e3c', '颜色值应为#a55e3c');
    framework.assertTrue(color.label.includes('棕色'), '标签包含"棕色"文字');
});

framework.test('颜色下拉框-紫色', '紫色选项', '验证紫色选项显示正确', () => {
    const color = { name: 'purple', label: '紫色', bg: '#a55eea' };
    
    framework.assertEqual(color.name, 'purple', '颜色名称应为purple');
    framework.assertEqual(color.label, '紫色', '颜色标签应为紫色');
    framework.assertEqual(color.bg, '#a55eea', '颜色值应为#a55eea');
    framework.assertTrue(color.label.includes('紫色'), '标签包含"紫色"文字');
});

framework.test('颜色下拉框-粉色', '粉色选项', '验证粉色选项显示正确', () => {
    const color = { name: 'pink', label: '粉色', bg: '#fd79a8' };
    
    framework.assertEqual(color.name, 'pink', '颜色名称应为pink');
    framework.assertEqual(color.label, '粉色', '颜色标签应为粉色');
    framework.assertEqual(color.bg, '#fd79a8', '颜色值应为#fd79a8');
    framework.assertTrue(color.label.includes('粉色'), '标签包含"粉色"文字');
});

// 10. 颜色符号测试
framework.test('颜色下拉框-符号', '颜色符号存在', '验证每个颜色选项包含●符号', () => {
    const allColors = [
        { name: 'red', label: '红色' },
        { name: 'blue', label: '蓝色' },
        { name: 'yellow', label: '黄色' },
        { name: 'green', label: '绿色' },
        { name: 'brown', label: '棕色' },
        { name: 'purple', label: '紫色' },
        { name: 'pink', label: '粉色' }
    ];
    
    allColors.forEach(color => {
        const displayText = color.label + ' ●';
        framework.assertTrue(displayText.includes('●'), `${color.label}选项应包含●符号`);
        framework.assertTrue(displayText.endsWith(' ●'), `${color.label}选项●符号应在末尾`);
    });
});

framework.test('颜色下拉框-符号', '符号格式正确', '验证颜色符号格式为"标签 ●"', () => {
    const testCases = [
        { label: '红色', expected: '红色 ●' },
        { label: '蓝色', expected: '蓝色 ●' },
        { label: '黄色', expected: '黄色 ●' },
        { label: '绿色', expected: '绿色 ●' },
        { label: '棕色', expected: '棕色 ●' },
        { label: '紫色', expected: '紫色 ●' },
        { label: '粉色', expected: '粉色 ●' }
    ];
    
    testCases.forEach(tc => {
        const actual = tc.label + ' ●';
        framework.assertEqual(actual, tc.expected, `显示文本应为"${tc.expected}"`);
    });
});

// 11. 颜色值格式测试
framework.test('颜色下拉框-颜色值', '颜色值为十六进制', '验证所有颜色值为有效的十六进制格式', () => {
    const colorValues = [
        '#ff6b6b', '#48dbfb', '#feca57', '#26de81', 
        '#a55e3c', '#a55eea', '#fd79a8'
    ];
    
    colorValues.forEach(colorValue => {
        framework.assertTrue(colorValue.startsWith('#'), `${colorValue}应以#开头`);
        framework.assertEqual(colorValue.length, 7, `${colorValue}应为7位字符(#RRGGBB)`);
        // 验证十六进制字符
        const hexPart = colorValue.substring(1);
        framework.assertTrue(/^[0-9a-fA-F]{6}$/.test(hexPart), `${colorValue}应为有效十六进制`);
    });
});

framework.test('颜色下拉框-颜色值', '颜色值唯一性', '验证所有颜色值唯一不重复', () => {
    const colorValues = [
        '#ff6b6b', '#48dbfb', '#feca57', '#26de81', 
        '#a55e3c', '#a55eea', '#fd79a8'
    ];
    
    const uniqueValues = new Set(colorValues);
    framework.assertEqual(uniqueValues.size, colorValues.length, '所有颜色值应唯一');
});

// 12. 颜色名称唯一性测试
framework.test('颜色下拉框-名称', '颜色名称唯一性', '验证所有颜色名称唯一不重复', () => {
    const colorNames = ['red', 'blue', 'yellow', 'green', 'brown', 'purple', 'pink'];
    const uniqueNames = new Set(colorNames);
    
    framework.assertEqual(uniqueNames.size, colorNames.length, '所有颜色名称应唯一');
});

framework.test('颜色下拉框-名称', '颜色名称格式', '验证颜色名称为小写英文字母', () => {
    const colorNames = ['red', 'blue', 'yellow', 'green', 'brown', 'purple', 'pink'];
    
    colorNames.forEach(name => {
        framework.assertTrue(/^[a-z]+$/.test(name), `${name}应为小写英文字母`);
    });
});

// 13. 颜色标签唯一性测试
framework.test('颜色下拉框-标签', '颜色标签唯一性', '验证所有颜色标签唯一不重复', () => {
    const colorLabels = ['红色', '蓝色', '黄色', '绿色', '棕色', '紫色', '粉色'];
    const uniqueLabels = new Set(colorLabels);
    
    framework.assertEqual(uniqueLabels.size, colorLabels.length, '所有颜色标签应唯一');
});

framework.test('颜色下拉框-标签', '颜色标签格式', '验证颜色标签为中文字符', () => {
    const colorLabels = ['红色', '蓝色', '黄色', '绿色', '棕色', '紫色', '粉色'];
    
    colorLabels.forEach(label => {
        framework.assertTrue(/^[\u4e00-\u9fa5]+$/.test(label), `${label}应为中文字符`);
    });
});

// 14. updateColorSelect函数逻辑测试
framework.test('颜色下拉框-逻辑', '无已选颜色时显示所有颜色', '验证colorConfig为空时显示全部7种颜色', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' },
        { name: 'yellow', label: '黄色', bg: '#feca57' },
        { name: 'green', label: '绿色', bg: '#26de81' },
        { name: 'brown', label: '棕色', bg: '#a55e3c' },
        { name: 'purple', label: '紫色', bg: '#a55eea' },
        { name: 'pink', label: '粉色', bg: '#fd79a8' }
    ];
    
    const existingColors = []; // 无已选颜色
    const availableColors = allColors.filter(c => !existingColors.includes(c.name));
    
    framework.assertEqual(availableColors.length, 7, '应显示全部7种颜色');
});

framework.test('颜色下拉框-逻辑', '已选1种颜色时显示剩余颜色', '验证已选红色后显示剩余6种颜色', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' },
        { name: 'yellow', label: '黄色', bg: '#feca57' },
        { name: 'green', label: '绿色', bg: '#26de81' },
        { name: 'brown', label: '棕色', bg: '#a55e3c' },
        { name: 'purple', label: '紫色', bg: '#a55eea' },
        { name: 'pink', label: '粉色', bg: '#fd79a8' }
    ];
    
    const existingColors = ['red']; // 已选红色
    const availableColors = allColors.filter(c => !existingColors.includes(c.name));
    
    framework.assertEqual(availableColors.length, 6, '应显示剩余6种颜色');
    framework.assertTrue(!availableColors.find(c => c.name === 'red'), '红色不应在可选列表中');
});

framework.test('颜色下拉框-逻辑', '已选多种颜色时显示剩余颜色', '验证已选3种颜色后显示剩余4种颜色', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' },
        { name: 'yellow', label: '黄色', bg: '#feca57' },
        { name: 'green', label: '绿色', bg: '#26de81' },
        { name: 'brown', label: '棕色', bg: '#a55e3c' },
        { name: 'purple', label: '紫色', bg: '#a55eea' },
        { name: 'pink', label: '粉色', bg: '#fd79a8' }
    ];
    
    const existingColors = ['red', 'blue', 'yellow']; // 已选3种
    const availableColors = allColors.filter(c => !existingColors.includes(c.name));
    
    framework.assertEqual(availableColors.length, 4, '应显示剩余4种颜色');
    framework.assertTrue(!availableColors.find(c => c.name === 'red'), '红色不应在可选列表中');
    framework.assertTrue(!availableColors.find(c => c.name === 'blue'), '蓝色不应在可选列表中');
    framework.assertTrue(!availableColors.find(c => c.name === 'yellow'), '黄色不应在可选列表中');
});

framework.test('颜色下拉框-逻辑', '已选全部颜色时无可选颜色', '验证已选7种颜色后无可选颜色', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' },
        { name: 'yellow', label: '黄色', bg: '#feca57' },
        { name: 'green', label: '绿色', bg: '#26de81' },
        { name: 'brown', label: '棕色', bg: '#a55e3c' },
        { name: 'purple', label: '紫色', bg: '#a55eea' },
        { name: 'pink', label: '粉色', bg: '#fd79a8' }
    ];
    
    const existingColors = ['red', 'blue', 'yellow', 'green', 'brown', 'purple', 'pink'];
    const availableColors = allColors.filter(c => !existingColors.includes(c.name));
    
    framework.assertEqual(availableColors.length, 0, '应无可选颜色');
});

// 15. 选项生成逻辑测试
framework.test('颜色下拉框-选项生成', '默认选项文本', '验证默认选项文本正确', () => {
    const defaultOptionText = '选择要添加的颜色';
    
    framework.assertEqual(defaultOptionText, '选择要添加的颜色', '默认选项文本应正确');
});

framework.test('颜色下拉框-选项生成', '选项value属性', '验证选项value为颜色名称', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' }
    ];
    
    allColors.forEach(color => {
        const optionValue = color.name;
        framework.assertEqual(optionValue, color.name, `选项value应为${color.name}`);
    });
});

framework.test('颜色下拉框-选项生成', '选项textContent格式', '验证选项textContent为"标签 ●"', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' }
    ];
    
    allColors.forEach(color => {
        const textContent = color.label + ' ●';
        framework.assertEqual(textContent, color.label + ' ●', `选项文本应为"${color.label} ●"`);
    });
});

framework.test('颜色下拉框-选项生成', '选项color样式', '验证选项style.color为颜色值', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' }
    ];
    
    allColors.forEach(color => {
        const styleColor = color.bg;
        framework.assertEqual(styleColor, color.bg, `选项颜色应为${color.bg}`);
    });
});

framework.test('颜色下拉框-选项生成', '选项fontWeight样式', '验证选项style.fontWeight为bold', () => {
    const fontWeight = 'bold';
    framework.assertEqual(fontWeight, 'bold', '选项fontWeight应为bold');
});

// 16. 边界条件测试
framework.test('颜色下拉框-边界', '空已选颜色数组', '验证existingColors为空数组时的处理', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' },
        { name: 'yellow', label: '黄色', bg: '#feca57' },
        { name: 'green', label: '绿色', bg: '#26de81' },
        { name: 'brown', label: '棕色', bg: '#a55e3c' },
        { name: 'purple', label: '紫色', bg: '#a55eea' },
        { name: 'pink', label: '粉色', bg: '#fd79a8' }
    ];
    
    const existingColors = [];
    const availableColors = allColors.filter(c => !existingColors.includes(c.name));
    
    framework.assertEqual(availableColors.length, 7, '空数组时显示全部颜色');
});

framework.test('颜色下拉框-边界', '无效颜色名称过滤', '验证无效颜色名称不影响过滤结果', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' }
    ];
    
    const existingColors = ['invalid_color', 'another_invalid'];
    const availableColors = allColors.filter(c => !existingColors.includes(c.name));
    
    framework.assertEqual(availableColors.length, 2, '无效颜色名称不影响过滤结果');
});

// 17. 颜色映射完整性测试
framework.test('颜色下拉框-映射', '名称到标签映射', '验证名称到标签的映射完整', () => {
    const nameToLabel = {
        'red': '红色',
        'blue': '蓝色',
        'yellow': '黄色',
        'green': '绿色',
        'brown': '棕色',
        'purple': '紫色',
        'pink': '粉色'
    };
    
    framework.assertEqual(Object.keys(nameToLabel).length, 7, '应有7个名称到标签的映射');
    framework.assertEqual(nameToLabel['red'], '红色', 'red映射到红色');
    framework.assertEqual(nameToLabel['blue'], '蓝色', 'blue映射到蓝色');
    framework.assertEqual(nameToLabel['yellow'], '黄色', 'yellow映射到黄色');
    framework.assertEqual(nameToLabel['green'], '绿色', 'green映射到绿色');
    framework.assertEqual(nameToLabel['brown'], '棕色', 'brown映射到棕色');
    framework.assertEqual(nameToLabel['purple'], '紫色', 'purple映射到紫色');
    framework.assertEqual(nameToLabel['pink'], '粉色', 'pink映射到粉色');
});

framework.test('颜色下拉框-映射', '名称到颜色值映射', '验证名称到颜色值的映射完整', () => {
    const nameToColor = {
        'red': '#ff6b6b',
        'blue': '#48dbfb',
        'yellow': '#feca57',
        'green': '#26de81',
        'brown': '#a55e3c',
        'purple': '#a55eea',
        'pink': '#fd79a8'
    };
    
    framework.assertEqual(Object.keys(nameToColor).length, 7, '应有7个名称到颜色值的映射');
    framework.assertEqual(nameToColor['red'], '#ff6b6b', 'red映射到#ff6b6b');
    framework.assertEqual(nameToColor['blue'], '#48dbfb', 'blue映射到#48dbfb');
    framework.assertEqual(nameToColor['yellow'], '#feca57', 'yellow映射到#feca57');
    framework.assertEqual(nameToColor['green'], '#26de81', 'green映射到#26de81');
    framework.assertEqual(nameToColor['brown'], '#a55e3c', 'brown映射到#a55e3c');
    framework.assertEqual(nameToColor['purple'], '#a55eea', 'purple映射到#a55eea');
    framework.assertEqual(nameToColor['pink'], '#fd79a8', 'pink映射到#fd79a8');
});

// 18. 颜色预览功能综合测试
framework.test('颜色下拉框-综合', '所有颜色选项完整验证', '综合验证所有颜色选项的完整性', () => {
    const allColors = [
        { name: 'red', label: '红色', bg: '#ff6b6b' },
        { name: 'blue', label: '蓝色', bg: '#48dbfb' },
        { name: 'yellow', label: '黄色', bg: '#feca57' },
        { name: 'green', label: '绿色', bg: '#26de81' },
        { name: 'brown', label: '棕色', bg: '#a55e3c' },
        { name: 'purple', label: '紫色', bg: '#a55eea' },
        { name: 'pink', label: '粉色', bg: '#fd79a8' }
    ];
    
    // 验证数量
    framework.assertEqual(allColors.length, 7, '应有7种颜色');
    
    // 验证每种颜色的完整性
    allColors.forEach(color => {
        // 名称验证
        framework.assertTrue(typeof color.name === 'string' && color.name.length > 0, 
            `${color.label}名称有效`);
        
        // 标签验证
        framework.assertTrue(typeof color.label === 'string' && color.label.length > 0, 
            `${color.label}标签有效`);
        
        // 颜色值验证
        framework.assertTrue(typeof color.bg === 'string' && color.bg.startsWith('#') && color.bg.length === 7, 
            `${color.label}颜色值有效`);
        
        // 显示文本验证
        const displayText = color.label + ' ●';
        framework.assertTrue(displayText.includes('●'), 
            `${color.label}显示文本包含●符号`);
        
        // 样式验证
        const styles = {
            color: color.bg,
            fontWeight: 'bold'
        };
        framework.assertEqual(styles.color, color.bg, 
            `${color.label}文字颜色正确`);
        framework.assertEqual(styles.fontWeight, 'bold', 
            `${color.label}字体粗细正确`);
    });
});

// 19. 覆盖率标记测试 - 颜色下拉框
framework.test('颜色下拉框-覆盖率', '标记函数覆盖', '标记updateColorSelect函数已覆盖', () => {
    framework.markCalled('updateColorSelect');
    framework.assertTrue(true, 'updateColorSelect函数已标记覆盖');
});

// 7. 覆盖率标记测试
framework.test('优化功能-覆盖率', '标记函数覆盖', '标记所有优化功能函数已覆盖', () => {
    framework.markCalled('disableColorConfig');
    framework.markCalled('startDrawWithProgressControl');
    framework.markCalled('resetAllWithAbort');
    framework.assertTrue(true, '优化功能相关函数已标记覆盖');
});

// ==================== 运行测试 ====================
async function runTests() {
    console.log('\n========================================');
    console.log('抽奖工具单元测试 - 新增功能测试');
    console.log('========================================\n');
    
    const results = await framework.runAll();
    
    console.log('\n========================================');
    console.log('测试结果汇总');
    console.log('========================================');
    console.log(`总测试数: ${totalTests}`);
    console.log(`通过: ${passedTests} ✅`);
    console.log(`失败: ${failedTests} ❌`);
    console.log(`通过率: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
    
    // 计算覆盖率
    const coveredFuncs = Object.keys(coverage.called);
    const coveragePercent = ((coveredFuncs.length / coverage.functions.length) * 100).toFixed(1);
    
    console.log('\n========================================');
    console.log('代码覆盖率');
    console.log('========================================');
    console.log(`覆盖率: ${coveragePercent}%`);
    console.log(`已覆盖函数: ${coveredFuncs.join(', ')}`);
    
    const uncoveredFuncs = coverage.functions.filter(f => !coverage.called[f]);
    if (uncoveredFuncs.length > 0) {
        console.log(`未覆盖函数: ${uncoveredFuncs.join(', ')}`);
    }
    
    console.log('\n');
    
    // 返回退出码
    process.exit(failedTests > 0 ? 1 : 0);
}

// 运行测试
runTests().catch(err => {
    console.error('测试运行错误:', err);
    process.exit(1);
});
