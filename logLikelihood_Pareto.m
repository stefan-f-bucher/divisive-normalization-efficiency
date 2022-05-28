% X is m-by-n where m is the number of data points and n the number of dimensions
function llh = logLikelihood_Pareto(X, mu, sigma, beta,n)
    X = reshape(X, length(X)/n, n);
    %n = size(X,2);
    llh = 0;
    for i=1:size(X,1)
        delta = log(0.25*Pareto_pdf(X(i,:), mu, sigma, beta, n));
        if isinf(delta)
            delta=-100000000;
        end
        llh = llh + delta;
    end
end