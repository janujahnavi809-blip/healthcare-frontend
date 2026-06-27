# Performance Optimization Report

## Optimizations Implemented

1. Measured dataset loading and model training times.
2. Added error handling for empty and large datasets.
3. Implemented model caching using Joblib to avoid retraining on every execution.
4. Reduced memory usage by closing charts after generation.
5. Reused the loaded dataset throughout the application.

## Result

The application now executes faster, uses memory efficiently, and is more stable when handling larger datasets.
