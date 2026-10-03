# Fréchet distance between Gaussians (FID core)

Implement `frechet_distance(mu1, S1, mu2, S2)` =
`||mu1 − mu2||² + Tr(S1 + S2 − 2·(S1^{1/2} S2 S1^{1/2})^{1/2})` for symmetric PSD covariances.

Compute matrix square roots through **symmetric eigendecompositions** (clip tiny negative eigenvalues to 0);
don't use `scipy.linalg.sqrtm`. Also implement `stats(features)` returning `(mean, covariance)` with the
`n − 1` denominator.
