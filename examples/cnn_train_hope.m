function [net, info] = cnn_train_hope(net, imdb, getBatch, varargin)
% CNN_TRAIN_HOPE: developed based on cnn_train in matconvnet (by Andrea Vedaldi)
% Do network training
% The hope training method is implemented in the function 'map_gradients'
% Data augmentation methods are implemented in the function 'process_epoch'

opts.batchSize = 100 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.gpus = 1 ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.beta = 0.015;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.momentum_hopefast = 0.9;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.isDropInput = 1;
opts.dropInputRate = 0.1;
opts.hopeMethod = 1; % 0: original; 1: new matrix1; 2: new matrix2
opts.isDataAug = 1;

opts = vl_argparse(opts, varargin) ;


if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
%save net.mat net;
if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
    % Legacy code: will be removed
    if isfield(net.layers{i}, 'filters')
      net.layers{i}.momentum{1} = zeros(size(net.layers{i}.filters), 'single') ;
      net.layers{i}.momentum{2} = zeros(size(net.layers{i}.biases), 'single') ;
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, 2, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = single([1 0]) ;
      end
    end
  end
end
%save net2.mat net;
% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

%save net3.mat net;
% setup error calculation function
if isstr(opts.errorFunction)
  switch opts.errorFunction
    case 'none'
      opts.errorFunction = @error_none ;
    case 'multiclass'
      opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1e', 'top5e'} ; end
    case 'binary'
      opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'bine'} ; end
    otherwise
      error('Uknown error function ''%s''', opts.errorFunction) ;
  end
end
% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

all_corr_collection = [];
for epoch=1:opts.numEpochs
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  beta = opts.beta(min(epoch, numel(opts.beta))) ;

  % fast-forward to last checkpoint
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;
  if opts.continue
    if exist(modelPath(epoch),'file')
      if epoch == opts.numEpochs
        load(modelPath(epoch), 'net', 'info') ;
      end
      continue ;
    end
    if epoch > 1
      fprintf('resuming by loading epoch %d\n', epoch-1) ;
      load(modelPath(epoch-1), 'net', 'info') ;
    end
  end
  
  % move CNN to GPU as needed
  if numGpus == 1
    net = vl_simplenn_move(net, 'gpu') ;
  elseif numGpus > 1
    spmd(numGpus)
      net_ = vl_simplenn_move(net, 'gpu') ;
    end
  end
  
  % train one epoch and validate
  train = opts.train(randperm(numel(opts.train))) ; % shuffle
  val = opts.val ;
  if numGpus <= 1
	[net,stats.train,all_corr] = process_epoch(opts, getBatch, epoch, train, learningRate, beta, imdb, net) ;
    [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 0, 0, imdb, net) ;
  else
    spmd(numGpus)
      [net_, stats_train_,all_corr] = process_epoch(opts, getBatch, epoch, train, learningRate, beta, imdb, net_) ;
      [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 0, 0, imdb, net_) ;
    end
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
  end
  all_corr_collection = [all_corr_collection, all_corr];

  % save
  if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  for f = sets
    f = char(f) ;
    n = numel(eval(f)) ;
    info.(f).speed(epoch) = n / stats.(f)(1) ;
    info.(f).objective(epoch) = stats.(f)(2) / n ;
    info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
  end
  if numGpus > 1
    spmd(numGpus)
      net_ = vl_simplenn_move(net_, 'cpu') ;
    end
    net = net_{1} ;
  else
    net = vl_simplenn_move(net, 'cpu') ;
  end
  if ~evaluateMode
	if (mod(epoch, 100) == 0)
		save(modelPath(epoch), 'net', 'info') ;
	end
  end

  figure(1) ; clf ;
  hasError = isa(opts.errorFunction, 'function_handle') ;
  hasHOPE = 1;
  subplot(1,1+hasError+hasHOPE,1) ;
  if ~evaluateMode
    plot(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
    hold on ;
  end
  plot(1:epoch, info.val.objective, '.--') ;
  xlabel('training epoch') ; ylabel('energy') ;
  grid on ;
  h=legend(sets) ;
  set(h,'color','none');
  title('objective') ;
  if hasError
    subplot(1,2+hasHOPE,2) ; leg = {} ;
    if ~evaluateMode
      plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
      hold on ;
      leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
    end
    plot(1:epoch, info.val.error', '.--') ;
    leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
    set(legend(leg{:}),'color','none') ;
    grid on ;
    xlabel('training epoch') ; ylabel('error') ;
    title('error') ;
  end
  if hasHOPE
    subplot(1,3,3) ; leg = {} ;
	nHOPE = size(all_corr_collection, 1);
    if ~evaluateMode
		for iHOPE = 1:nHOPE
			plot(1:epoch, all_corr_collection(iHOPE, :), '.-', 'linewidth', 2) ;
			hold on ;
			leg = horzcat(leg, strcat('HOPE ', iHOPE)) ;
		end
    end
    set(legend(leg{:}),'color','none') ;
    grid on ;
    xlabel('training epoch') ; ylabel('Correlation') ;
    title('Correlation') ;
  end
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
[~,predictions] = sort(predictions, 3, 'descend') ;
error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
err(1,1) = sum(sum(sum(error(:,:,1,:)))) ;
err(2,1) = sum(sum(sum(min(error(:,:,1:5,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_binaryclass(opts, labels, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = sum(error(:)) ;

% -------------------------------------------------------------------------
function err = error_none(opts, labels, res)
% -------------------------------------------------------------------------
err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [net,stats,all_corr,prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, beta, imdb, net)
% -------------------------------------------------------------------------

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 3, mpiprofile on ; end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
res = [] ;
mmap = [] ;
stats = [] ;

top_1_Error_collection = zeros(1, 2);
for t=1:opts.batchSize:numel(subset)
  fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
  batchTime = tic ;
  numDone = 0 ;
  error = [] ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
	[im, labels] = getBatch(imdb, batch) ;

    if opts.prefetch
      if s==opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      getBatch(imdb, nextBatch) ;
    end
	
	% Do horizontal flip for 50% images
		if training
		if opts.isDataAug % && (epoch < (opts.numEpochs-100) )
			fprintf('Do image augmentation (rotation+translation+scale+colorspace)... ');
			[H0, W0, C0, N0] = size(im);
			randIdx = randperm(N0); % translation
			randIdx2 = randperm(N0); % rotate
			randIdx3 = randperm(N0); % scale
			randIdx4 = randperm(N0); % RGB casting
			for iN = 1:N0
				if (randIdx(iN) <= (N0/2))
					transRangeX = (rand(1) - 0.5) * 10;
					transRangeY = (rand(1) - 0.5) * 10;
					im(:, :, :, iN) = imtranslate(im(:, :, :, iN),[transRangeX, transRangeY]);
				end
				
				if (randIdx2(iN) <= (N0/2))
					angle = (rand(1) - 0.5) * 10; % -5 ~ 5
					im(:, :, :, iN) = imrotate( im(:, :, :, iN), angle, 'bilinear', 'crop' );
				end
				
				if (randIdx3(iN) <= (N0/2))
					xmin = randperm(4);
					ymin = randperm(4);
					width = randperm(4) + 24;
					height = randperm(4) + 24;
					tmpIm = imcrop(im(:, :, :, iN), [xmin(1), ymin(1), width(1), height(1)]);
					im(:, :, :, iN) = imresize(tmpIm, [32, 32]);
				end
				
				if (randIdx4(iN) <= (N0/2))
					tagR = rand(1);
					tagG = rand(1);
					tagB = rand(1);
					% [H0, W0, C0, N0] = size(im);
					maskRGB = ( rand(H0, W0, C0, N0)*0.1 - 0.05 ) + 1;% 0-1 -> 0.95-1.05
					if (tagR < 0.5)
						im(:, :, 1, iN) = im(:, :, 1, iN) .* maskRGB(:, :, 1, iN);
					end
					if (tagG < 0.5)
						im(:, :, 2, iN) = im(:, :, 2, iN) .* maskRGB(:, :, 2, iN);
					end				
					if (tagB < 0.5)
						im(:, :, 3, iN) = im(:, :, 3, iN) .* maskRGB(:, :, 3, iN);
					end
				end
			end	
			fprintf('Done ... \n');
		end
	end
  
	
	if opts.isDropInput
		if training
			fprintf('Do drop input ');
			[H, W, C, N] = size(im);
			mask = single(rand(H, W) >= opts.dropInputRate) ;
			im = bsxfun(@times, im, mask);
		else
			% im = im * (1 - opts.dropInputRate);
		end
	end

    if numGpus >= 1
      im = gpuArray(im) ;
    end
	
    % evaluate CNN
    net.layers{end}.class = labels ;
    if training, dzdy = one; else, dzdy = [] ; end
	%'accumulate', s~=1, ...
    res = vl_simplenn(net, im, dzdy, res, ...
                      'accumulate', 0, ...
                      'disableDropout', ~training, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync) ;

    % accumulate training errors	
    error = sum([error, [...
      sum(double(gather(res(end).x))) ;
      reshape(opts.errorFunction(opts, labels, res),[],1) ; ]],2) ;
    numDone = numDone + numel(batch) ;
  end

  % gather and accumulate gradients across labs
  all_corr = 0;
  if training
    if numGpus <= 1
      [net, all_corr] = accumulate_gradients(opts, learningRate, beta, batchSize, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net, all_corr, res] = accumulate_gradients(opts, learningRate, beta, batchSize, net, res, mmap) ;
    end
  end

  % print learning statistics
  batchTime = toc(batchTime) ;
  stats = sum([stats,[batchTime ; error]],2); % works even when stats=[]
  speed = batchSize/batchTime ;

  fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
  n = (t + batchSize - 1) / max(1,numlabs) ;
  fprintf(' obj:%.6g', stats(2)/n) ;
  for i=1:numel(opts.errorLabels)
    fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/n) ;
	top_1_Error_collection(i) = top_1_Error_collection(i) + stats(i+2)/n;
  end
  fprintf(' [%d/%d]', numDone, batchSize);
  fprintf('\n') ;

  % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
  end
end

top_1_Error_collection = top_1_Error_collection ./ numel(subset);
top_1_Error_collection = top_1_Error_collection .* opts.batchSize;
fprintf(' Overall top-1 Error: %.6g, overall top-5 Error: %.6g \n', top_1_Error_collection(1), top_1_Error_collection(2)) ;

if nargout > 3
  prof = mpiprofile('info');
  mpiprofile off ;
end

% -------------------------------------------------------------------------
function [net,all_corr,res] = accumulate_gradients(opts, lr, beta, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
nHOPE = 0;
all_corr = [];
for l=1:numel(net.layers)
	for j=1:numel(res(l).dzdw)
		if nargin >= 7
			tag = sprintf('l%d_%d',l,j) ;
			tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
			for g = setdiff(1:numel(mmap.Data), labindex)
				tmp = tmp + mmap.Data(g).(tag) ;
			end
			res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
		end
	end

    if isfield(net.layers{l}, 'weights')
	 thisDecay(1) = opts.weightDecay * net.layers{l}.weightDecay(1) ;
     thisLR(1) = lr * net.layers{l}.learningRate(1) ;
	 thisDecay(2) = opts.weightDecay * net.layers{l}.weightDecay(2) ;
     thisLR(2) = lr * net.layers{l}.learningRate(2) ;
	
	 if ( strcmp(net.layers{l}.type, 'hope_fast') )
		nHOPE = nHOPE + 1;
		if (opts.hopeMethod == 0) % slow
			L2_W = sqrt(sum(net.layers{l}.weights{1}.^2, 1)); %[1, W, C_in, C_out]
			%df_D = zeros(H, W, C_in, C_out, 'single');
			%df_D = gpuArray(df_D);
			for i2 = 1:net.layers{l}.C_in
				for j2 = 1:net.layers{l}.C_out
					weight = net.layers{l}.weights{1}(:, :, i2, j2);% can be optimized
					% L2_W_S = L2_W_Square(:, :, i2, j2);
					L2_W_sub = L2_W(:, :, i2, j2);
					L2_W_S = transpose(L2_W_sub)*L2_W_sub;
					C_matrix = transpose(weight) * weight;
					G_matrix = C_matrix./L2_W_S;
					B = sign(C_matrix)./L2_W_S;
					% df_D(:, :, i2, j2) = weight*B - (weight./repmat(L2_W_sub.^2,[size(weight,1),1]))* diag(sum(G_matrix,1));
					net.layers{l}.df_D(:, :, i2, j2) = weight*B - (bsxfun(@rdivide, weight, L2_W_sub.^2))* diag(sum(G_matrix,1));
				end
			end
		elseif (opts.hopeMethod == 1) % this method that mentioned in our paper
			weight = reshape(net.layers{l}.weights{1}, [], net.layers{l}.C_out);
			L2_W = sqrt(sum(weight.^2, 1)); %[1, C_out]
			L2_W_S = transpose(L2_W)*L2_W; % C_out * C_out
			C_matrix = transpose(weight) * weight; % C_out * C_out
			G_matrix = C_matrix./L2_W_S;
			B = sign(C_matrix)./L2_W_S;
			net.layers{l}.df_D = weight*B - (bsxfun(@rdivide, weight, L2_W.^2))* diag(sum(G_matrix,1));
			
			net.layers{l}.df_D = reshape(net.layers{l}.df_D, net.layers{l}.H, net.layers{l}.W, net.layers{l}.C_in, net.layers{l}.C_out);
		else
			for i = 1:net.layers{l}.C_out
				weight = net.layers{l}.weights{1}(:, :, :, i);% can be optimized
				weight = reshape(weight, [], 1);
				L2_W = sqrt(sum(weight.^2, 1));
				L2_W_S = transpose(L2_W)*L2_W;
				C_matrix = transpose(weight) * weight;
				G_matrix = C_matrix./L2_W_S;
				B = sign(C_matrix)./L2_W_S;
							
				df_D0 = weight*B - (bsxfun(@rdivide, weight, L2_W.^2))* diag(sum(G_matrix,1));
				net.layers{l}.df_D(:, :, :, i) = reshape(df_D0, net.layers{l}.H, net.layers{l}.W, net.layers{l}.C_in, 1);
			end
		end
		res(l).dzdw{1} = res(l).dzdw{1} + beta .* net.layers{l}.df_D;
	    
		net.layers{l}.momentum{1} = ...
			opts.momentum_hopefast * net.layers{l}.momentum{1} ...
			- thisLR(1) * thisDecay(1) * net.layers{l}.weights{1} ...
			- thisLR(1) * (1 / batchSize) * res(l).dzdw{1} ; 
		
		net.layers{l}.weights{1} = net.layers{l}.weights{1} + net.layers{l}.momentum{1} ;	
		
		if (opts.hopeMethod == 0)
			%[H, W, C_in, C_out] = size(net.layers{l}.weights{1});	
			L2_W = sqrt(sum(net.layers{l}.weights{1}.^2,1));%[1, W, C_in, C_out]
			net.layers{l}.weights{1} = bsxfun(@rdivide, net.layers{l}.weights{1}, L2_W);
		
			correlation = sum(sum(abs(transpose(net.layers{l}.weights{1}(:, :, 1, 1) )*net.layers{l}.weights{1}(:, :, 1, 1) ),1),2)-size(net.layers{l}.weights{1}(:, :, 1, 1),2);
		elseif (opts.hopeMethod == 1)
			weight = reshape(net.layers{l}.weights{1}, [], net.layers{l}.C_out);
			L2_W = sqrt(sum(weight.^2, 1)); %[1, C_out]
			weight = bsxfun(@rdivide, weight, L2_W);
			net.layers{l}.weights{1} = reshape(weight, net.layers{l}.H, net.layers{l}.W, net.layers{l}.C_in, net.layers{l}.C_out);
			
			correlation = sum(sum(abs(transpose(weight)*weight),1),2)-net.layers{l}.C_out;
		else
			L2_W = sqrt(sum(net.layers{l}.weights{1}.^2,1));%[1, W, C_in, C_out]
			net.layers{l}.weights{1} = bsxfun(@rdivide, net.layers{l}.weights{1}, L2_W);
		
			weight2 = net.layers{l}.weights{1}(:, :, :, 1);
			weight2 = reshape(weight2, [], 1);
			
			correlation = sum(sum(abs(transpose(weight2)*weight2),1),2)-1;
		end
		
		fprintf(' corr %.4f, hopeMethod %d ', correlation, opts.hopeMethod);
		all_corr(nHOPE, 1) = gather(correlation);
		% clear L2_W; 
	 end
	  
	  if ( strcmp(net.layers{l}.type, 'conv') ) 
		net.layers{l}.momentum{1} = ...
			opts.momentum * net.layers{l}.momentum{1} ...
			- thisLR(1) * thisDecay(1) * net.layers{l}.weights{1} ...
			- thisLR(1) * (1 / batchSize) * res(l).dzdw{1} ; 
		
		net.layers{l}.weights{1} = net.layers{l}.weights{1} + net.layers{l}.momentum{1} ;
	  
		net.layers{l}.momentum{2} = ...
			opts.momentum * net.layers{l}.momentum{2} ...
			- thisLR(2) * thisDecay(2) * net.layers{l}.weights{2} ...
			- thisLR(2) * (1 / batchSize) * res(l).dzdw{2} ; 
		
		net.layers{l}.weights{2} = net.layers{l}.weights{2} + net.layers{l}.momentum{2} ;
	  end
	  
	  if ( strcmp(net.layers{l}.type, 'bnorm') ) 
		net.layers{l}.momentum{1} = ...
			opts.momentum * net.layers{l}.momentum{1} ...
			- thisLR(1) * thisDecay(1) * net.layers{l}.weights{1} ...
			- thisLR(1) * (1 / batchSize) * res(l).dzdw{1} ; 
		
		net.layers{l}.weights{1} = net.layers{l}.weights{1} + net.layers{l}.momentum{1} ;
	  
		net.layers{l}.momentum{2} = ...
			opts.momentum * net.layers{l}.momentum{2} ...
			- thisLR(2) * thisDecay(2) * net.layers{l}.weights{2} ...
			- thisLR(2) * (1 / batchSize) * res(l).dzdw{2} ; 
		
		net.layers{l}.weights{2} = net.layers{l}.weights{2} + net.layers{l}.momentum{2} ;
	  end
	  
    end
  %end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end
