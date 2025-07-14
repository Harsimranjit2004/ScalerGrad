import math
import random
import numpy as np
from typing import List, Tuple

def test_scalar_operations():
    """Test basic scalar operations and their gradients."""
    print("=" * 50)
    print("TESTING SCALAR OPERATIONS")
    print("=" * 50)
    
    # Test 1: Addition
    print("\n1. Testing Addition:")
    x = Scalar(3.0, label='x')
    y = Scalar(4.0, label='y')
    z = x + y
    z.backward()
    print(f"x + y = {z.data} (expected: 7.0)")
    print(f"dx = {x.grad} (expected: 1.0)")
    print(f"dy = {y.grad} (expected: 1.0)")
    assert abs(z.data - 7.0) < 1e-6
    assert abs(x.grad - 1.0) < 1e-6
    assert abs(y.grad - 1.0) < 1e-6
    print("✓ Addition test passed!")
    
    # Test 2: Multiplication
    print("\n2. Testing Multiplication:")
    x = Scalar(3.0, label='x')
    y = Scalar(4.0, label='y')
    z = x * y
    z.backward()
    print(f"x * y = {z.data} (expected: 12.0)")
    print(f"dx = {x.grad} (expected: 4.0)")
    print(f"dy = {y.grad} (expected: 3.0)")
    assert abs(z.data - 12.0) < 1e-6
    assert abs(x.grad - 4.0) < 1e-6
    assert abs(y.grad - 3.0) < 1e-6
    print("✓ Multiplication test passed!")
    
    # Test 3: Power
    print("\n3. Testing Power:")
    x = Scalar(2.0, label='x')
    z = x ** 3
    z.backward()
    print(f"x^3 = {z.data} (expected: 8.0)")
    print(f"dx = {x.grad} (expected: 12.0)")  # 3 * x^2 = 3 * 4 = 12
    assert abs(z.data - 8.0) < 1e-6
    assert abs(x.grad - 12.0) < 1e-6
    print("✓ Power test passed!")
    
    # Test 4: Chain rule
    print("\n4. Testing Chain Rule:")
    x = Scalar(2.0, label='x')
    y = x * 3
    z = y + 1
    w = z ** 2
    w.backward()
    print(f"w = ((x * 3) + 1)^2 = {w.data} (expected: 49.0)")
    print(f"dw/dx = {x.grad} (expected: 42.0)")  # 2 * (6+1) * 3 = 42
    assert abs(w.data - 49.0) < 1e-6
    assert abs(x.grad - 42.0) < 1e-6
    print("✓ Chain rule test passed!")

def test_activation_functions():
    """Test activation functions and their gradients."""
    print("\n" + "=" * 50)
    print("TESTING ACTIVATION FUNCTIONS")
    print("=" * 50)
    
    # Test 1: ReLU
    print("\n1. Testing ReLU:")
    x1 = Scalar(2.0, label='x1')
    x2 = Scalar(-1.0, label='x2')
    y1 = x1.relu()
    y2 = x2.relu()
    y1.backward()
    y2.backward()
    print(f"ReLU(2.0) = {y1.data} (expected: 2.0)")
    print(f"ReLU(-1.0) = {y2.data} (expected: 0.0)")
    print(f"d(ReLU(2.0))/dx = {x1.grad} (expected: 1.0)")
    print(f"d(ReLU(-1.0))/dx = {x2.grad} (expected: 0.0)")
    assert abs(y1.data - 2.0) < 1e-6
    assert abs(y2.data - 0.0) < 1e-6
    assert abs(x1.grad - 1.0) < 1e-6
    assert abs(x2.grad - 0.0) < 1e-6
    print("✓ ReLU test passed!")
    
    # Test 2: Tanh
    print("\n2. Testing Tanh:")
    x = Scalar(0.0, label='x')
    y = x.tanh()
    y.backward()
    print(f"tanh(0.0) = {y.data} (expected: 0.0)")
    print(f"d(tanh(0.0))/dx = {x.grad} (expected: 1.0)")
    assert abs(y.data - 0.0) < 1e-6
    assert abs(x.grad - 1.0) < 1e-6
    print("✓ Tanh test passed!")
    
    # Test 3: Sigmoid
    print("\n3. Testing Sigmoid:")
    x = Scalar(0.0, label='x')
    y = x.sigmoid()
    y.backward()
    print(f"sigmoid(0.0) = {y.data} (expected: 0.5)")
    print(f"d(sigmoid(0.0))/dx = {x.grad} (expected: 0.25)")
    assert abs(y.data - 0.5) < 1e-6
    assert abs(x.grad - 0.25) < 1e-6
    print("✓ Sigmoid test passed!")

def test_mathematical_functions():
    """Test mathematical functions."""
    print("\n" + "=" * 50)
    print("TESTING MATHEMATICAL FUNCTIONS")
    print("=" * 50)
    
    # Test 1: Exponential
    print("\n1. Testing Exponential:")
    x = Scalar(1.0, label='x')
    y = x.exp()
    y.backward()
    expected = math.e
    print(f"exp(1.0) = {y.data} (expected: {expected})")
    print(f"d(exp(1.0))/dx = {x.grad} (expected: {expected})")
    assert abs(y.data - expected) < 1e-6
    assert abs(x.grad - expected) < 1e-6
    print("✓ Exponential test passed!")
    
    # Test 2: Logarithm
    print("\n2. Testing Logarithm:")
    x = Scalar(math.e, label='x')
    y = x.log()
    y.backward()
    print(f"log(e) = {y.data} (expected: 1.0)")
    print(f"d(log(e))/dx = {x.grad} (expected: {1/math.e})")
    assert abs(y.data - 1.0) < 1e-6
    assert abs(x.grad - 1/math.e) < 1e-6
    print("✓ Logarithm test passed!")
    
    # Test 3: Sine
    print("\n3. Testing Sine:")
    x = Scalar(math.pi/2, label='x')
    y = x.sin()
    y.backward()
    print(f"sin(π/2) = {y.data} (expected: 1.0)")
    print(f"d(sin(π/2))/dx = {x.grad} (expected: 0.0)")
    assert abs(y.data - 1.0) < 1e-6
    assert abs(x.grad - 0.0) < 1e-6
    print("✓ Sine test passed!")
    
    # Test 4: Cosine
    print("\n4. Testing Cosine:")
    x = Scalar(0.0, label='x')
    y = x.cos()
    y.backward()
    print(f"cos(0.0) = {y.data} (expected: 1.0)")
    print(f"d(cos(0.0))/dx = {x.grad} (expected: 0.0)")
    assert abs(y.data - 1.0) < 1e-6
    assert abs(x.grad - 0.0) < 1e-6
    print("✓ Cosine test passed!")

def test_tensor_operations():
    """Test tensor operations."""
    print("\n" + "=" * 50)
    print("TESTING TENSOR OPERATIONS")
    print("=" * 50)
    
    # Test 1: 1D tensor addition
    print("\n1. Testing 1D Tensor Addition:")
    t1 = Scalar([1.0, 2.0, 3.0], label='t1')
    t2 = Scalar([4.0, 5.0, 6.0], label='t2')
    t3 = t1 + t2
    t3.backward()
    print(f"t1 + t2 = {t3.data} (expected: [5.0, 7.0, 9.0])")
    print(f"dt1 = {t1.grad} (expected: [1.0, 1.0, 1.0])")
    print(f"dt2 = {t2.grad} (expected: [1.0, 1.0, 1.0])")
    assert t3.data == [5.0, 7.0, 9.0]
    assert t1.grad == [1.0, 1.0, 1.0]
    assert t2.grad == [1.0, 1.0, 1.0]
    print("✓ 1D tensor addition test passed!")
    
    # Test 2: 1D tensor multiplication
    print("\n2. Testing 1D Tensor Multiplication:")
    t1 = Scalar([2.0, 3.0, 4.0], label='t1')
    t2 = Scalar([5.0, 6.0, 7.0], label='t2')
    t3 = t1 * t2
    t3.backward()
    print(f"t1 * t2 = {t3.data} (expected: [10.0, 18.0, 28.0])")
    print(f"dt1 = {t1.grad} (expected: [5.0, 6.0, 7.0])")
    print(f"dt2 = {t2.grad} (expected: [2.0, 3.0, 4.0])")
    assert t3.data == [10.0, 18.0, 28.0]
    assert t1.grad == [5.0, 6.0, 7.0]
    assert t2.grad == [2.0, 3.0, 4.0]
    print("✓ 1D tensor multiplication test passed!")
    
    # Test 3: Dot product
    print("\n3. Testing Dot Product:")
    v1 = Scalar([1.0, 2.0, 3.0], label='v1')
    v2 = Scalar([4.0, 5.0, 6.0], label='v2')
    result = v1.dot(v2)
    result.backward()
    expected = 1*4 + 2*5 + 3*6  # 32
    print(f"v1 · v2 = {result.data} (expected: {expected})")
    print(f"dv1 = {v1.grad} (expected: [4.0, 5.0, 6.0])")
    print(f"dv2 = {v2.grad} (expected: [1.0, 2.0, 3.0])")
    assert abs(result.data - expected) < 1e-6
    assert v1.grad == [4.0, 5.0, 6.0]
    assert v2.grad == [1.0, 2.0, 3.0]
    print("✓ Dot product test passed!")

def test_broadcasting():
    """Test broadcasting between scalars and tensors."""
    print("\n" + "=" * 50)
    print("TESTING BROADCASTING")
    print("=" * 50)
    
    # Test 1: Scalar + Tensor
    print("\n1. Testing Scalar + Tensor:")
    s = Scalar(2.0, label='s')
    t = Scalar([1.0, 2.0, 3.0], label='t')
    result = s + t
    result.backward()
    print(f"scalar + tensor = {result.data} (expected: [3.0, 4.0, 5.0])")
    print(f"ds = {s.grad} (expected: 3.0)")  # Sum of gradients
    print(f"dt = {t.grad} (expected: [1.0, 1.0, 1.0])")
    assert result.data == [3.0, 4.0, 5.0]
    assert abs(s.grad - 3.0) < 1e-6
    assert t.grad == [1.0, 1.0, 1.0]
    print("✓ Scalar + Tensor test passed!")
    
    # Test 2: Scalar * Tensor
    print("\n2. Testing Scalar * Tensor:")
    s = Scalar(3.0, label='s')
    t = Scalar([2.0, 4.0, 6.0], label='t')
    result = s * t
    result.backward()
    print(f"scalar * tensor = {result.data} (expected: [6.0, 12.0, 18.0])")
    print(f"ds = {s.grad} (expected: 12.0)")  # Sum of tensor values
    print(f"dt = {t.grad} (expected: [3.0, 3.0, 3.0])")
    assert result.data == [6.0, 12.0, 18.0]
    assert abs(s.grad - 12.0) < 1e-6
    assert t.grad == [3.0, 3.0, 3.0]
    print("✓ Scalar * Tensor test passed!")

def test_neural_network_example():
    """Test a simple neural network example."""
    print("\n" + "=" * 50)
    print("TESTING NEURAL NETWORK EXAMPLE")
    print("=" * 50)
    
    # Simple 2-layer neural network
    print("\nSimple 2-layer Neural Network:")
    
    # Input
    x = Scalar([1.0, 2.0], label='x')
    
    # Layer 1 weights and bias
    w1 = Scalar([[0.1, 0.2], [0.3, 0.4]], label='w1')
    b1 = Scalar([0.5, 0.6], label='b1')
    
    # Layer 1 forward pass: h = tanh(x · w1 + b1)
    # For simplicity, we'll do element-wise operations
    h1_pre = Scalar([
        x.data[0] * w1.data[0][0] + x.data[1] * w1.data[0][1],
        x.data[0] * w1.data[1][0] + x.data[1] * w1.data[1][1]
    ], label='h1_pre')
    
    h1 = (h1_pre + b1).tanh()
    
    # Layer 2 weights and bias
    w2 = Scalar([0.7, 0.8], label='w2')
    b2 = Scalar(0.9, label='b2')
    
    # Layer 2 forward pass: output = h · w2 + b2
    output = h1.dot(w2) + b2
    
    print(f"Neural network output: {output.data}")
    
    # Backward pass
    output.backward()
    
    print(f"Gradient w.r.t. w2: {w2.grad}")
    print(f"Gradient w.r.t. b2: {b2.grad}")
    
    print("✓ Neural network example completed!")

def test_gradient_checking():
    """Test numerical gradient checking."""
    print("\n" + "=" * 50)
    print("TESTING GRADIENT CHECKING")
    print("=" * 50)
    
    def numerical_gradient(f, x, h=1e-7):
        """Compute numerical gradient using finite differences."""
        if x.shape == ():
            x_plus = Scalar(x.data + h)
            x_minus = Scalar(x.data - h)
            return (f(x_plus).data - f(x_minus).data) / (2 * h)
        else:
            grad = [0.0] * len(x.data)
            for i in range(len(x.data)):
                x_plus_data = x.data[:]
                x_minus_data = x.data[:]
                x_plus_data[i] += h
                x_minus_data[i] -= h
                x_plus = Scalar(x_plus_data)
                x_minus = Scalar(x_minus_data)
                grad[i] = (f(x_plus).data - f(x_minus).data) / (2 * h)
            return grad
    
    # Test 1: Simple function
    print("\n1. Testing f(x) = x^2:")
    x = Scalar(3.0, label='x')
    def f(x): return x ** 2
    
    result = f(x)
    result.backward()
    
    analytical_grad = x.grad
    numerical_grad = numerical_gradient(f, Scalar(3.0))
    
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient: {numerical_grad}")
    print(f"Difference: {abs(analytical_grad - numerical_grad)}")
    
    assert abs(analytical_grad - numerical_grad) < 1e-5
    print("✓ Gradient checking test passed!")

def test_complex_expressions():
    """Test complex mathematical expressions."""
    print("\n" + "=" * 50)
    print("TESTING COMPLEX EXPRESSIONS")
    print("=" * 50)
    
    # Test 1: f(x, y) = sin(x) * cos(y) + exp(x * y)
    print("\n1. Testing f(x, y) = sin(x) * cos(y) + exp(x * y):")
    x = Scalar(1.0, label='x')
    y = Scalar(0.5, label='y')
    
    f = x.sin() * y.cos() + (x * y).exp()
    f.backward()
    
    print(f"f(1.0, 0.5) = {f.data}")
    print(f"df/dx = {x.grad}")
    print(f"df/dy = {y.grad}")
    
    # Manual calculation for verification
    expected_f = math.sin(1.0) * math.cos(0.5) + math.exp(1.0 * 0.5)
    expected_dx = math.cos(1.0) * math.cos(0.5) + 0.5 * math.exp(0.5)
    expected_dy = math.sin(1.0) * (-math.sin(0.5)) + 1.0 * math.exp(0.5)
    
    print(f"Expected f = {expected_f}")
    print(f"Expected df/dx = {expected_dx}")
    print(f"Expected df/dy = {expected_dy}")
    
    assert abs(f.data - expected_f) < 1e-6
    assert abs(x.grad - expected_dx) < 1e-6
    assert abs(y.grad - expected_dy) < 1e-6
    print("✓ Complex expression test passed!")

def run_all_tests():
    """Run all tests."""
    print("STARTING COMPREHENSIVE AUTODIFF TESTS")
    print("=" * 60)
    
    try:
        test_scalar_operations()
        test_activation_functions()
        test_mathematical_functions()
        test_tensor_operations()
        test_broadcasting()
        test_neural_network_example()
        test_gradient_checking()
        test_complex_expressions()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Your autodiff implementation is working correctly!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Make sure to import the Scalar class from your implementation
    # from your_scalar_module import Scalar
    
    run_all_tests()