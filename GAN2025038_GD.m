function GAN20250501_EnhancedBackprop
    %Enhanced version of Yuto's program with additional backpropagation layer
    %Fixed random numbers for result comparison
    
    BigT=10; %number of time steps
    BigN=5;  %number of firms
    BigD=2;  %number of moment conditions = number of neurons in
            %third layer of maximization network
    n2=2;    %number of neurons in second layer of both networks
    n1=2;    %number of input neurons
    
    %%%% DATA %%%%%%%%%%
    x1=[0.02  0.02  0.02  0.02  0.02;
    0.0195  0.0195  0.0195  0.0195  0.0195;
    0.019  0.019  0.019  0.019  0.019;
    0.0185  0.0185  0.0185  0.0185  0.0185;
    0.018  0.018  0.018  0.018  0.018;
    0.0175  0.0175  0.0175  0.0175  0.0175;
    0.017  0.017  0.017  0.017  0.017;
    0.0165  0.0165  0.0165  0.0165  0.0165;
    0.016  0.016  0.016  0.016  0.016;
    0.0155  0.0155  0.0155  0.0155  0.0155]; %market return
    
    x2=[72  77  88  71  44;
    59  66  101  62  78;
    65  70  79  38  33;
    44  47  112  25  22;
    40  49  88  54  12;
    51  55  86  50  45;
    44  47  55  53  24;
    39  44  69  43  12;
    68  72  65  70  74;
    71  75  74  70  72]; %weather
    
    Rexcess=[0.03  0.031  0.032  0.035  0.04;
    0.01  0.0305  0.0315  0.0325  0.0255;
    -0.01  0.03  0.031  0.0144  0.07;
    0.035  0.0295  0.0144  0.0315  0.06;
    0.028  0.029  0.03  0.035  0.022;
    0.015  0.0285  0.0295  0.0305  0.0315;
    0.07  0.028  0.0144  0.03  0.2;
    0.0265  0.0275  0.0285  0.0295  0.08;
    0.026  0.027  0.028  0.029  0.05;
    0.0144  0.0265  0.0144  0.035  0.049];
    
    %Normalizing step
    x1=x1/0.02;
    x2=x2/72;
    
    % Additional variables for the new layer
    % Parameters for the input transformation layer
    phi1 = 0.5;
    phi2 = 0.5;
    c1 = 0.2;
    c2 = 0.2;
    s1 = 1.0;
    s2 = 1.0;
    
    % Store previous timestep data for recurrent connections
    x_prev = zeros(BigT, BigN, n1);
    
    rng(5000);
    x=[x1(1,1); x2(1,1)];
    
    % Network parameters
    W2g=[1 0; 0 1];
    W3g=[1 0; 0 1];
    b2g=[0;0];
    b3g=[0;0];
    
    W3o=[1 1];
    W2o=[1 0; 0 1];
    b2o=[0;0];
    b3o=0;
    
    eta=2;             %learning rate
    Niter=10;          %iterations
    maxiter=2;
    
    savecostmin=zeros(Niter,1);
    savecostmax=zeros(Niter,1);
    savecost=zeros(Niter,2);
    omega=zeros(BigT, BigN,1);
    
    % Initialize storage for layer activations
    a1o = zeros(BigT, BigN, n1);  % New first layer output
    a2o = zeros(BigT, BigN, n2);
    a3o = zeros(BigT, BigN, 1);
    
    % Initialize storage for transformed inputs
    xi_t = zeros(BigT, BigN, n1);
    
    for counter=1:Niter
        % Forward pass with additional layer
        for t=1:BigT
            for i=1:BigN
                raw_x = [x1(t,i); x2(t,i)];
                
                % Get previous timestep data (with boundary handling)
                if t > 1
                    x_prev_t = [x1(t-1,i); x2(t-1,i)];
                else
                    x_prev_t = [0; 0]; % Initial value for first timestep
                end
                x_prev(t,i,:) = x_prev_t;
                
                % Apply input transformation using the parameters from derivatives
                xi_t1 = (raw_x(1) - phi1*x_prev_t(1))/s1;
                xi_t2 = (raw_x(2) - phi2*x_prev_t(2))/s2;
                xi_t(t,i,:) = [xi_t1; xi_t2];
                
                % Store transformed input as first layer activation
                a1o(t,i,:) = [xi_t1; xi_t2];
                
                % Continue with the rest of the network
                a2otemp = activate([xi_t1; xi_t2], W2o, b2o);
                a2o(t,i,:) = a2otemp;
                a3o(t,i) = activate(a2otemp, W3o, b3o);
                omega(t,i) = a3o(t,i);
            end
        end
        
        % Backpropagation
        delta3o = a3o.*(1-a3o);  % Gradient for sigmoid activation
        delta2o = a2o.*(1-a2o);
        delta1o = zeros(BigT, BigN, n1);  % Gradient for the new layer
        
        % Compute deltas for all layers
        for t=1:BigT
            for i=1:BigN
                % Backpropagate from layer 3 to layer 2
                for Index_n2=1:n2
                    delta2o(t,i,Index_n2) = delta2o(t,i,Index_n2)*(W3o(Index_n2)*delta3o(t,i));
                end
                
                % Backpropagate from layer 2 to layer 1 (new)
                for Index_n1=1:n1
                    delta1o(t,i,Index_n1) = 0;
                    for Index_n2=1:n2
                        delta1o(t,i,Index_n1) = delta1o(t,i,Index_n1) + delta2o(t,i,Index_n2)*W2o(Index_n2,Index_n1);
                    end
                    delta1o(t,i,Index_n1) = delta1o(t,i,Index_n1) * (1/s1); % Adjust by scale factor
                end
                
                % Compute gradients for W2o and W3o (original params)
                gradient_omega(t,i,1) = delta2o(t,i,1)*xi_t(t,i,1);  % dW2o_11
                gradient_omega(t,i,2) = delta2o(t,i,1)*xi_t(t,i,2);  % dW2o_12
                gradient_omega(t,i,3) = delta2o(t,i,2)*xi_t(t,i,1);  % dW2o_21
                gradient_omega(t,i,4) = delta2o(t,i,2)*xi_t(t,i,2);  % dW2o_22
                gradient_omega(t,i,5) = delta2o(t,i,1);  % db2o_1
                gradient_omega(t,i,6) = delta2o(t,i,2);  % db2o_2
                gradient_omega(t,i,7) = delta3o(t,i)*a2o(t,i,1);  % dW3o_1
                gradient_omega(t,i,8) = delta3o(t,i)*a2o(t,i,2);  % dW3o_2
                gradient_omega(t,i,9) = delta3o(t,i);  % db3o
                
                % New gradients for phi, c, and s parameters
                % Based on the derivatives in the image
                if t > 1
                    % dphi1
                    gradient_omega(t,i,10) = delta1o(t,i,1) * (-x_prev(t,i,1)/s1);
                    % dphi2
                    gradient_omega(t,i,11) = delta1o(t,i,2) * (-x_prev(t,i,2)/s2);
                    % dc1 - using appropriate formula from image
                    gradient_omega(t,i,12) = delta1o(t,i,1) * (-1/s1);
                    % dc2
                    gradient_omega(t,i,13) = delta1o(t,i,2) * (-1/s2);
                    % ds1
                    gradient_omega(t,i,14) = delta1o(t,i,1) * (-xi_t(t,i,1)/s1);
                    % ds2
                    gradient_omega(t,i,15) = delta1o(t,i,2) * (-xi_t(t,i,2)/s2);
                else
                    % Zero gradients for first timestep where no previous data exists
                    gradient_omega(t,i,10:15) = 0;
                end
                
                % Gradients for input at previous timestep (for completeness)
                if t > 1
                    % dx_{t-1,1}
                    gradient_omega(t,i,16) = delta1o(t,i,1) * (-phi1/s1);
                    % dx_{t-1,2}
                    gradient_omega(t,i,17) = delta1o(t,i,2) * (-phi2/s2);
                else
                    gradient_omega(t,i,16:17) = 0;
                end
            end
        end
        
        % Batch gradient computation
        gradient_Sjdsq = zeros(1, 17);  % Expanded to include new parameters
        
        for j=1:BigN
            for d=1:BigD
                temp_gradient = zeros(1,17);
                for t=1:BigT
                    Sum_i_grad_om = zeros(1,17);
                    for i=1:BigN
                        for coefftheta=1:17
                            Sum_i_grad_om(coefftheta) = Sum_i_grad_om(coefftheta) + gradient_omega(t,i,coefftheta)*Rexcess(t,i);
                        end
                    end
                    for coefftheta=1:17
                        temp_gradient(coefftheta) = temp_gradient(coefftheta) + Rexcess(t,j)*Sum_i_grad_om(coefftheta);
                    end
                end
                temp_gradient = -fun_s_jd(Rexcess, omega, j, d) * temp_gradient;
                gradient_Sjdsq = gradient_Sjdsq + temp_gradient;
            end
        end
        
        % Update all parameters
        W2o = W2o - eta*reshape(gradient_Sjdsq(1:4), 2, 2);
        b2o = b2o - eta*gradient_Sjdsq(5:6)';
        W3o = W3o - eta*gradient_Sjdsq(7:8);
        b3o = b3o - eta*gradient_Sjdsq(9);
        
        % Update new parameters
        phi1 = phi1 - eta*gradient_Sjdsq(10);
        phi2 = phi2 - eta*gradient_Sjdsq(11);
        c1 = c1 - eta*gradient_Sjdsq(12);
        c2 = c2 - eta*gradient_Sjdsq(13);
        s1 = s1 - eta*gradient_Sjdsq(14);
        s2 = s2 - eta*gradient_Sjdsq(15);
        
        % Compute objective function
        newcost = 0;
        for j=1:BigN
            newcost = newcost + fun_s_j(Rexcess, omega, j);
        end
        savecostmin(counter) = newcost;
        
        % Display parameter values for monitoring
        fprintf('Iteration %d: Cost = %f\n', counter, newcost);
        fprintf('  phi1=%f, phi2=%f, c1=%f, c2=%f, s1=%f, s2=%f\n', phi1, phi2, c1, c2, s1, s2);
    end
    
    % Display final results
    disp('Final Cost Values:');
    disp(savecostmin);
    
    disp('Final Parameter Values:');
    fprintf('phi1=%f, phi2=%f\n', phi1, phi2);
    fprintf('c1=%f, c2=%f\n', c1, c2);
    fprintf('s1=%f, s2=%f\n', s1, s2);
    fprintf('W2o=\n');
    disp(W2o);
    fprintf('b2o=\n');
    disp(b2o);
    fprintf('W3o=\n');
    disp(W3o);
    fprintf('b3o=%f\n', b3o);
end

% Helper functions
function s_j_val=fun_s_j(Rexcess, omega, j)
    s_j_val=0;
    for dd=1:2
        s_j_val=s_j_val+fun_s_jd(Rexcess, omega, j, dd);
    end
end

function s_jd_val=fun_s_jd(Rexcess, omega, j, d)
    s_jd_val=0;
    for tt=1:10
        portfret=0;
        for ii=1:5
            omega_ti=omega(tt,ii);
            portfret=portfret+omega_ti*Rexcess(tt,ii);
        end
        s_jd_val=s_jd_val+(1-portfret)*Rexcess(tt,j);
    end
end

function y=activate(x, W, b)
    y=1./(1+exp(-(W*x+b)));
end