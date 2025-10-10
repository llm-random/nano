# Plasticity Scheduler Tests

Tests for the plasticity learning rate schedulers.

## Running Tests

From the project root directory:

```bash
# Run individual tests
python -m src.plasticity.tests.test_repeated_scheduler
python -m src.plasticity.tests.test_scheduler_final
python -m src.plasticity.tests.test_scheduler_comprehensive

# Or run all tests
python -m pytest src/plasticity/tests/ -v
```

## Test Files

### test_repeated_scheduler.py
Basic functionality tests for `RepeatedScheduler`:
- WSD scheduler with repeated cycles
- Cosine scheduler with repeated cycles
- Lambda factory function usage

### test_scheduler_final.py
Validation tests to ensure no negative learning rates across various configurations:
- Different cycle counts (2, 3, 5)
- Different step counts (500, 1000)
- Different warmup configurations

### test_scheduler_comprehensive.py
Comprehensive behavioral tests:
- Cycle boundary verification
- Warmup override functionality
- No warmup in subsequent cycles
- Multiple scheduler types (WSD, Cosine)

## Expected Output

All tests should pass with output like:
```
✓ PASS: Test description
```

Any failures will show:
```
✗ FAIL: Error description
```
