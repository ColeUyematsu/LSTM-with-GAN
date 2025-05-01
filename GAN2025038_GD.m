function GAN20250328_BatchGD_fromYuto
    %This is the program Yuto sent me
    %In this program random numbers are fixed, so you can compare your results
    %with mine
    BigT=10; %number of time steps
    BigN=5;  %number of firms
    BigD=2;  %number of moment conditions = number of neurons in
            %third layer of maximization network
    n2=2;    %number of neurons in second layer of both networks
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
    rng(5000);
    x=[x1(1,1); x2(1,1)];
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
    %for iter=1:maxiter %no need to flip flop between min and max
    %   for t=1:BigT
    %       for j=1:BigN
    %           x=[x1(t,j);x2(t,j)];
    %           g(t,j,:)=fun_g(x,W2g,W3g,b2g,b3g);
    %       end
    %   end
        for counter=1:Niter  
           for t=1:BigT
               for i=1:BigN
                   x=[x1(t,i);x2(t,i)];
                   a2otemp=activate(x,W2o,b2o);
                   a2o(t,i,:)=a2otemp;
                   a3o(t,i)=activate(a2otemp,W3o,b3o);
                   omega(t,i)=a3o(t,i);
               end
           end
           delta3o=a3o.*(1-a3o);
           delta2o=a2o.*(1-a2o);
           for t=1:BigT
               for i=1:BigN
                   for Index_n2=1:n2
                       delta2o(t,i,Index_n2)=delta2o(t,i,Index_n2)*(W3o(Index_n2)*delta3o(t,i));
                   end
                   gradient_omega(t,i,1)= delta2o(t,i,1)*x1(t,i);
                   gradient_omega(t,i,2)= delta2o(t,i,1)*x2(t,i);
                   gradient_omega(t,i,3)= delta2o(t,i,2)*x1(t,i);
                   gradient_omega(t,i,4)= delta2o(t,i,2)*x2(t,i);
                   gradient_omega(t,i,5)= delta2o(t,i,1);
                   gradient_omega(t,i,6)= delta2o(t,i,2);
                   gradient_omega(t,i,7)= delta3o(t,i)*a2o(t,i,1);
                   gradient_omega(t,i,8)= delta3o(t,i)*a2o(t,i,2);
                   gradient_omega(t,i,9)= delta3o(t,i);
               end
           end
           gradient_Sjdsq = zeros(1, 9);  % Initialize gradient accumulator for batch mode (replaces single-sample update in SGD)  %%% FIXED initialization
           for j=1:BigN
               for d=1:BigD
                   temp_gradient = zeros(1,9);
                   for t=1:BigT
                       Sum_i_grad_om = zeros(1,9);
                       for i=1:BigN
                           for coefftheta=1:9
                               Sum_i_grad_om(coefftheta) = Sum_i_grad_om(coefftheta) + gradient_omega(t,i,coefftheta)*Rexcess(t,i);
                           end
                       end
                       for coefftheta=1:9
                           temp_gradient(coefftheta) = temp_gradient(coefftheta) + Rexcess(t,j)*Sum_i_grad_om(coefftheta);
                       end
                   end
                   temp_gradient = -fun_s_jd(Rexcess, x, j, d) * temp_gradient;
                   gradient_Sjdsq = gradient_Sjdsq + temp_gradient;  % Accumulate gradients across all (j,d) combinations instead of a random pair (replaces SGD with batch GD)
               end
           end
           gradient_Sjdsq = gradient_Sjdsq; % / (BigN * BigD); %%% CHANGED
           W2o=W2o - eta*reshape(gradient_Sjdsq(1:4),2,2);
           b2o=b2o - eta*gradient_Sjdsq(5:6)';
           W3o=W3o - eta*gradient_Sjdsq(7:8);
           b3o=b3o - eta*gradient_Sjdsq(9);
           newcost=0;
           for j=1:BigN
               newcost=newcost+fun_s_j(Rexcess, x,j);
           end
           savecostmin(counter)=newcost;
        end %counter
        savecostmin
    % code deleted after that
    %end -> flip flop between min and max
    
    function s_j_val=fun_s_j(Rexcess,x,j)
       s_j_val=0;
       for dd=1:2
           s_j_val=s_j_val+fun_s_jd(Rexcess, x, j,dd);
       end
    end
    function s_jd_val=fun_s_jd(Rexcess, x,j,d)
       s_jd_val=0;
       for tt=1:10
           portfret=0;
           for ii=1:5
               omega_ti=a3o(tt,ii);
               portfret=portfret+omega_ti*Rexcess(tt,ii);
           end
           s_jd_val=s_jd_val+(1-portfret)*Rexcess(tt,j);
       end
    end
    function omega_ti_val=fun_omega(x,W2o, W3o,b2o,b3o)
       a2o=activate(x,W2o,b2o);
       a3o=activate(a2o,W3o,b3o);
       omega_ti_val=a3o;
    end
    function g_val=fun_g(x,W2g, W3g,b2g,b3g)
       LOCALa2g=activate(x,W2g,b2g);
       LOCALa3g=activate(LOCALa2g,W3g,b3g);
       g_val=LOCALa3g;
    end
    function y=activate(x,W,b)
       y=1./(1+exp(-(W*x+b)));
    end
    end
    