%% Pareto pdf

% x is a vector (ONE multidimensional data point)
function density = Pareto_pdf(x, mu, sigma, beta, n)
    if all(x>mu)
        density = beta^n * factorial(n) * prod( (1./sigma) .* ( (x-mu)./sigma ).^(beta-1) ) / ( ( 1 + sum( ((x-mu)./sigma ).^beta ))^(n+1) );
    else
        density = 0;
    end
end
