UbitName = 'armaango';
personNumber = '50170093';

load project1_data.mat out_mat;
output_val = out_mat(:,1);
input_var = out_mat(:,[2:47]);

    %calculating percentages for data division
    training_percent = round((size(input_var,1)*0.8));
    validation_percent = round((size(input_var,1)*0.9));
    test_percent = round((size(input_var,1)*0.1));
    
    %dividing input data
    input_training_cfs = input_var(1:training_percent,:);%training data
    input_validation_cfs = input_var(training_percent+1:validation_percent,:);%validation data
    input_test_cfs = input_var(validation_percent+1:size(input_var,1) ,:);%test data
    %dividing output data
    output_training_cfs = output_val(1:training_percent,:);
    output_validation_cfs = output_val(training_percent+1:validation_percent,:);
    output_test_cfs = output_val(validation_percent+1:size(input_var,1) ,:);
    
    mu = [];
    mu_vec = [];
    fi_matrix_cfs = [];
    fi_matrix_cfs_validation = [];
    N =  size(input_var,1);
    N1 = size(input_training_cfs,1);
    N2 = size(input_validation_cfs,1);
    M1 = 10;                %Basis functions
    lambda1=0.03;
    
    
    sigma_vec = (var(input_var)+0.1);
    sigma_vec_final = diag(sigma_vec);
    
    %cond(sigma_vec_final)       %print on Command Window
    
    Sigma1=sigma_vec_final;
    for i=2:M1;
        Sigma1(:,:,i)=sigma_vec_final;      %DxDxM Sigma
    end
    
    for i=1:M1;
        input_var(1,:);
        mu = input_var((N-108*i),:);
        mu_vec=[mu_vec;mu];
        
        X_minus_mu = bsxfun(@minus,input_training_cfs,mu);
        mat = zeros(N1,1);
        if( i ==1)
            mat = ones(N1,1);
        else   
            for row= 1:N1
                mat(row,1) = exp(-(X_minus_mu(row,:) * inv(sigma_vec_final)* X_minus_mu(row,:)')/2);
            end  
        end
        fi_matrix_cfs = [fi_matrix_cfs mat];        %compute design matrix on training input data
    end
    
    mu1= mu_vec';
    
    training_size = size(fi_matrix_cfs,2);

    %calculate w using training set 
    
    w1 = pinv(lambda1*eye(training_size) + transpose(fi_matrix_cfs)*fi_matrix_cfs)*transpose(fi_matrix_cfs)*output_training_cfs;
    
    trainind1 = []; %making the training index matrix
    for s=1:training_percent
        trainind1(s)=s;
    end
    
    trainInd1 = trainind1';
    
    validind1 = []; %making the validation index matrix
    for f=1:size(input_validation_cfs,1)
        validind1(f)=f+55698;
    end
    
    validInd1 = validind1';
    validation_size = size(input_training_cfs,1);
    
    square_err = (sum(power(output_training_cfs - transpose((transpose(w1)*transpose(fi_matrix_cfs))),2)))/2;
    error = power((square_err*2)/validation_size,1/2);
    trainPer1 = error;
    
    
    %validPer1
    for i=1:M1;
        input_var(1,:);
        mu = input_var((N-108*i),:);
        mu_vec=[mu_vec;mu];
        
        X_minus_mu = bsxfun(@minus,input_validation_cfs,mu);
        mat = zeros(N2,1);
        if( i ==1)
            mat = ones(N2,1);
        else   
            for row= 1:N2
                mat(row,1) = exp(-(X_minus_mu(row,:) * inv(sigma_vec_final)* X_minus_mu(row,:)')/2);
            end  
        end
        fi_matrix_cfs_validation = [fi_matrix_cfs_validation mat];        %compute design matrix on validation input data
    end
    
    
    validation_size = size(input_validation_cfs,1);
    
    square_err = (sum(power(output_validation_cfs - transpose((transpose(w1)*transpose(fi_matrix_cfs_validation))),2)))/2;
    error = power((square_err*2)/validation_size,1/2);
    validPer1 = error;
    
    %----------gradient descent solution starts------------------------
    eta = 0.01;
    
    dw = [];
    dw1 = [];
    training_size_col = size(fi_matrix_cfs,2);
    training_size_rows = size(fi_matrix_cfs,1);
    validation_size=training_size_rows;
   
    W = zeros(training_size_col,1);
    w01=W;
    square_err = (sum(power(output_training_cfs - transpose((transpose(W)*transpose(fi_matrix_cfs))),2)))/2;
    error1 = power((square_err*2)/validation_size,1/2);
    for i=1:training_size_rows
     Y = output_training_cfs(i);
     dw=eta*((Y - transpose(W)*transpose(fi_matrix_cfs(i,:)))*transpose(fi_matrix_cfs(i,:))-((lambda1*W)/training_size_rows));
     W = W+dw;
     dw1=[dw1 dw];
     eta1(i)=eta;
     %modifying eta based on error
     square_err = (sum(power(output_training_cfs - transpose((transpose(W)*transpose(fi_matrix_cfs))),2)))/2;
     error2 = power((square_err*2)/validation_size,1/2);
     if error2>error1
         eta=eta/2;
     end
     if error2<error1
         eta=eta*1.01;
     end
     error1=error2;
    end
    
    diff1=norm(w1-W);

 %-------------------------------------Synthetic data operations--------------------------
 
load synthetic.mat x t;
output_val_syn = t;
input_var_syn = x';

    %calculating percentages for data division
    training_percent_syn = round((size(input_var_syn,1)*0.8));
    validation_percent_syn = round((size(input_var_syn,1)*1));
    
    %dividing input data
    input_training_cfs_syn = input_var_syn(1:training_percent_syn,:);
    input_validation_cfs_syn = input_var_syn(training_percent_syn+1:validation_percent_syn,:);
 
    %dividing output data
    output_training_cfs_syn = output_val_syn(1:training_percent_syn,:);
    output_validation_cfs_syn = output_val_syn(training_percent_syn+1:validation_percent_syn,:);
  
    
    mu_syn = [];
    mu_vec_syn = [];
    sigma_syn = [];
    fi_matrix_cfs_syn = [];
    fi_matrix_cfs_validation_syn=[];
    N =  size(input_var_syn,1);
    N1 = size(input_training_cfs_syn,1);
    M2 = 5;                %Basis functions
    lambda2=0.05;
    
    
    sigma_vec_syn = (var(input_var_syn)+0.1);
    sigma_vec_final = diag(sigma_vec_syn);
    
    
    Sigma2=sigma_vec_final;
    for i=2:M2;
        Sigma2(:,:,i)=sigma_vec_final;      %DxDxM Sigma
    end
    
    for i=1:M2;
        input_var_syn(1,:);
        mu_syn = input_var_syn((N-108*i),:);
        mu_vec_syn=[mu_vec_syn;mu_syn];
        
        X_minus_mu_syn = bsxfun(@minus,input_training_cfs_syn,mu_syn);
        mat = zeros(N1,1);
        if( i ==1)
            mat = ones(N1,1);
        else   
            for row= 1:N1
                mat(row,1) = exp(-(X_minus_mu_syn(row,:) * inv(sigma_vec_final)* X_minus_mu_syn(row,:)')/2);
            end  
        end
        fi_matrix_cfs_syn = [fi_matrix_cfs_syn mat];        %compute design matrix on training input data
    end
    
    mu2= mu_vec_syn';
    
    training_size = size(fi_matrix_cfs_syn,2);

    %calculate w using training set 
    
    w2 = pinv(lambda2*eye(training_size) + transpose(fi_matrix_cfs_syn)*fi_matrix_cfs_syn)*transpose(fi_matrix_cfs_syn)*output_training_cfs_syn;
    
    %creating trainInd and validInd Matrices
    trainind2 = [];
    for s=1:training_percent_syn
        trainind2(s)=s;
    end
    
    trainInd2 = trainind2';
    
    validind2 = [];
    for f=1:400
        validind2(f)=f+1600;
    end
    
    validInd2 = validind2';
    validation_size = size(input_training_cfs_syn,1);
    
    square_err = (sum(power(output_training_cfs_syn - transpose((transpose(w2)*transpose(fi_matrix_cfs_syn))),2)))/2;
    error = power((square_err*2)/validation_size,1/2);
    trainPer2 = error;
    
    
    
    
    %computing validPer2
    N2=size(input_validation_cfs_syn,1);
    for i=1:M2
        input_var_syn(1,:);
        mu_syn = input_var_syn((N-108*i),:);
        mu_vec_syn=[mu_vec_syn;mu_syn];
        
        X_minus_mu = bsxfun(@minus,input_validation_cfs_syn,mu_syn);
        mat = zeros(N2,1);
        if( i ==1)
            mat = ones(N2,1);
        else   
            for row= 1:N2
                mat(row,1) = exp(-(X_minus_mu(row,:) * inv(sigma_vec_final)* X_minus_mu(row,:)')/2);
            end  
        end
        fi_matrix_cfs_validation_syn = [fi_matrix_cfs_validation_syn mat];        %compute design matrix on validation input data
    end
    
    
    validation_size = size(input_validation_cfs_syn,1);
    
    square_err = (sum(power(output_validation_cfs_syn - transpose((transpose(w2)*transpose(fi_matrix_cfs_validation_syn))),2)))/2;
    error = power((square_err*2)/validation_size,1/2);
    validPer2 = error;
    
    
    
    
    %----------gradient descent solution-------------------------
    
    eta_syn = 0.008;
    dw = [];
    dw2 = [];
    training_size_col = size(fi_matrix_cfs_syn,2);
    training_size_rows = size(fi_matrix_cfs_syn,1);
    validation_size=training_size_rows;
   
    
    W = zeros(training_size_col,1);
    w02=W;
    
    square_err = (sum(power(output_training_cfs_syn - transpose((transpose(W)*transpose(fi_matrix_cfs_syn))),2)))/2;
    error1 = power((square_err*2)/validation_size,1/2);
    
    for i=1:training_size_rows
     Y = output_training_cfs_syn(i);
     dw= eta_syn*((Y - transpose(W)*transpose(fi_matrix_cfs_syn(i,:)))*transpose(fi_matrix_cfs_syn(i,:))-((lambda2*W)/training_size_rows));
     W = W + dw;
     dw2=[dw2 dw];
     eta2(i)=eta_syn;
     %modifying eta based on error
     square_err = (sum(power(output_training_cfs_syn - transpose((transpose(W)*transpose(fi_matrix_cfs_syn))),2)))/2;
     error2 = power((square_err*2)/validation_size,1/2);
     if error2>error1
         eta_syn=eta_syn/2;
     end
     error1=error2;
    end

    W_gd = W;
    diff2=norm(w2-W_gd);
    
    save('proj2.mat', 'UbitName', 'personNumber','M1', 'mu1', 'Sigma1', 'lambda1', 'w1', 'trainInd1', 'validInd1', 'trainPer1', 'w01', 'dw1', 'eta1', 'validPer1','M2', 'mu2', 'Sigma2', 'lambda2', 'w2', 'trainInd2', 'validInd2', 'trainPer2','validPer2','w02','dw2','eta2');
   