%% Pareto pdf

% x is a vector of length n (ONE multidimensional data point)
% mu is a vector of length n
% sigma is a vector of length n
% betaexponent is a scalar
% extendedToR2 is a boolean: 1 if the extension of the Pareto distribution to R^2 should be used
function density = Pareto_pdf(x, mu, sigma, betaexponent, extendedToR2)
    n = length(x);
    assert(length(mu)==length(x), 'mu must be of the same length as x');
    assert(length(sigma)==length(x), 'sigma must be of the same length as x');
    paretoPDF = @(x, mu, sigma, betaexponent, n) betaexponent^n * factorial(n) * prod( (1./sigma) .* ( (x-mu)./sigma ).^(betaexponent-1) ) / ( ( 1 + sum( ((x-mu)./sigma ).^betaexponent ))^(n+1) ); % eq.7
    if not(extendedToR2)
        if all(x > mu)
            density = paretoPDF(x, mu, sigma, betaexponent, n); 
        else
            density = 0;
        end 
    else % extend Pareto to R^2: f(s; mu=0,sigma,beta) = 0.25*f(|s|; mu=0,sigma,beta)
        assert(all(mu==0), 'extension to R^2 is only implemented for mu=0');
        if any(x==0)
            warning('Pareto dist. extended to R^2 evaluated to 0 at s_i=0')
        end
        density = 0.25 * paretoPDF(abs(x), mu, sigma, betaexponent, n);
    end
end
