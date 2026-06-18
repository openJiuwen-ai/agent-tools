"""线程安全的负载存储测试"""

import threading
import time

from openjiuwentools.infer_router.worker.atomic_load_store import AtomicLoadStore


def test_atomic_load_store_concurrent():
    """测试并发读写安全性"""
    store = AtomicLoadStore()
    worker_id = "test-worker"
    iterations = 1000

    def writer():
        for i in range(iterations):
            store.set_load(worker_id, float(i))
            time.sleep(0.0001)

    def reader(results):
        for _ in range(iterations):
            results.append(store.get_load(worker_id))
            time.sleep(0.0001)

    # 创建多个读写线程
    results = []
    threads = []

    for _ in range(3):
        t = threading.Thread(target=writer)
        threads.append(t)

    for _ in range(5):
        t = threading.Thread(target=reader, args=(results,))
        threads.append(t)

    # 启动所有线程
    for t in threads:
        t.start()

    # 等待完成
    for t in threads:
        t.join()

    # 验证结果有效性
    assert len(results) == 5 * iterations
    for val in results:
        assert isinstance(val, float)


def test_atomic_load_store_update():
    """测试原子更新操作"""
    store = AtomicLoadStore()
    worker_id = "test-worker"

    # 初始值为0
    assert store.get_load(worker_id) == 0.0

    # 设置初始值
    store.set_load(worker_id, 10.0)
    assert store.get_load(worker_id) == 10.0

    # 使用 update_load 进行原子更新
    new_val = store.update_load(worker_id, lambda x: x * 2 + 1)
    assert new_val == 21.0
    assert store.get_load(worker_id) == 21.0


def test_atomic_load_store_remove():
    """测试移除负载记录"""
    store = AtomicLoadStore()
    worker_id = "test-worker"

    store.set_load(worker_id, 50.0)
    assert store.get_load(worker_id) == 50.0

    store.remove_load(worker_id)
    assert store.get_load(worker_id) == 0.0


def test_atomic_load_store_get_all():
    """测试获取所有负载快照"""
    store = AtomicLoadStore()

    store.set_load("worker1", 10.0)
    store.set_load("worker2", 20.0)
    store.set_load("worker3", 30.0)

    snapshot = store.get_all_loads()
    assert isinstance(snapshot, dict)
    assert len(snapshot) == 3
    assert snapshot["worker1"] == 10.0
    assert snapshot["worker2"] == 20.0
    assert snapshot["worker3"] == 30.0

    # 修改快照不影响原存储
    snapshot["worker1"] = 100.0
    assert store.get_load("worker1") == 10.0


def test_atomic_load_store_concurrent_update():
    """测试并发更新操作"""
    store = AtomicLoadStore()
    worker_id = "test-worker"
    store.set_load(worker_id, 0.0)

    iterations = 1000

    def incrementer():
        for _ in range(iterations):
            store.update_load(worker_id, lambda x: x + 1)

    # 创建多个线程同时递增
    threads = []
    for _ in range(10):
        t = threading.Thread(target=incrementer)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # 预期结果: 10 * 1000 = 10000
    assert store.get_load(worker_id) == 10000.0
