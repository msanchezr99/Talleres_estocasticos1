% Parámetros de la simulación
num_trayectorias = 10000;
num_observaciones = 252;
dt = 1/252;

% Inicialización de las trayectorias para las tres simulaciones
X1 = zeros(num_trayectorias, num_observaciones+1);
X2 = zeros(num_trayectorias, num_observaciones+1);
X3 = zeros(num_trayectorias, num_observaciones+1);

% Definimos el punto inicial
X1(:,1) = 1;
X2(:,1) = 1;
X3(:,1) = 1;

% Valores de mu y sigma
mu_values = [rand(), rand(), rand()];
sigma_values = [0.4*rand(), 0.4*rand(), 0.4*rand()];

% Movimiento browniano
incrementos = sqrt(dt) * randn(num_trayectorias, num_observaciones);
B_t = cumsum([zeros(num_trayectorias, 1), incrementos], 2);

% Ecuación diferencial estocástica lineal homogénea
for j = 1:3
    mu = mu_values(j);
    sigma = sigma_values(j);
    
    for i = 2:num_observaciones+1
        if j == 1
            X1(:,i) = X1(:,i-1) + mu*X1(:,i-1)*dt + sigma*X1(:,i-1).*(B_t(:,i) - B_t(:,i-1));
        elseif j == 2
            X2(:,i) = X2(:,i-1) + mu*X2(:,i-1)*dt + sigma*X2(:,i-1).*(B_t(:,i) - B_t(:,i-1));
        else
            X3(:,i) = X3(:,i-1) + mu*X3(:,i-1)*dt + sigma*X3(:,i-1).*(B_t(:,i) - B_t(:,i-1));
        end
    end
end

% Graficar los resultados
figure;
plot(X1');
xlabel('Tiempo');
title('EDS Lineal Homogénea');
legend(['Simulación 1: \mu = ' num2str(mu_values(1)) ', \sigma = ' num2str(sigma_values(1))])

% Análisis Longitudinal
trayectoria_L= X1(1, :);

% Distribución
figure;
histfit(trayectoria_L, 50, '');
title('Distribución de una Trayectoria');

% Autocorrelación Parcial
figure;
parcorr(trayectoria_L);
title('Autocorrelación Parcial de una Trayectoria');

% Dimensión fractal exponente de Hurst, define como debe ser el



% Análisis Transversal (usando el último punto temporal)
trayectorias_T_final = X1(:, end);

% Distribución  
figure;
histfit(trayectorias_T, 50, 'lognormal');
title('Distribución Transversal en el Tiempo Final');
