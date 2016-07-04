function [net, info] = cnn_cifar_hope(varargin)
%  ------------  CNN Cifar-10 ------------------------
%  iFLYTEK Laboratory for Neural Computing and Machine Learning (iNCML) @ York University
%  (https://wiki.eecs.yorku.ca/lab/MLL/start)
%  Based on MatconvNet (http://www.vlfeat.org/matconvnet/)
%  Coding by: Hengyue Pan (panhy@cse.yorku.ca), York University, Toronto
%
%  If you hope to use this code, please cite:
% @article{pan2016learning,
%  title={Learning Convolutional Neural Networks using Hybrid Orthogonal Projection and Estimation},
%  author={Pan, Hengyue and Jiang, Hui},
%  journal={arXiv preprint arXiv:1606.05929},
%  year={2016}
%  }
%  License: MIT License
%
%  Using HOPE model to improve the CNN performance
%  Without data augmentation: 7.57% error rate on the validation set

run(fullfile(fileparts(mfilename('fullpath')), '../matlab/vl_setupnn.m')) ;

opts.modelType = 'HOPE' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 400 ;

switch opts.modelType
  case 'HOPE'
    lr_schedule = zeros(1, opts.train.numEpochs);
	beta_schedule = zeros(1, opts.train.numEpochs);
	lr_init = 0.06;
	beta_init = 0.15; % the learning rate of the orthogonal penalty term
	for i = 1:opts.train.numEpochs
		if (mod(i, 25) == 0)
			lr_init = lr_init / 2;
			beta_init = beta_init / 1.75;
		end
		lr_schedule(i) = lr_init;
		beta_schedule(i) = beta_init;
	end
	opts.train.learningRate = lr_schedule;
	opts.train.beta = beta_schedule;
	
    opts.train.weightDecay = 0.0005 ;
  otherwise
    error('Unknown model type %s', opts.modelType) ;
end

opts.expDir = fullfile('data','cifar-hope') ; % output folder
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','cifar') ;
opts.dataClass = 0; % RGB: 0; YUV: 1
opts.whitenData = false ; % if whitening
opts.contrastNormalization = false ; 

if (opts.dataClass == 0)
	opts.imdbPath = '/eecs/research/asr/hengyue/DFT_DNN/Data/cifar/imdb_cifar.mat';
	% imdb data structure:
	% imdb.images.data: 32-by-32-by-3-by-60000 (all cifar data)
	% imdb.images.labels: 1-by-60000 (all labels)
	% imdb.images.set: 1-by-60000: if the ith image is an training sample, then imdb.images.set(i) = 1, otherwise imdb.images.set(i) = 3
else
	opts.imdbPath = '/eecs/research/asr/hengyue/dataset/imdb_YUV_normalized.mat';
end
opts.train.continue = false ;
opts.train.gpus = 1 ;
opts.train.expDir = opts.expDir ;

opts.train.momentum = 0.9 ;
opts.train.isDropInput = 0;
opts.train.dropInputRate = 0.0;
opts.train.hopeMethod = 1; % 2, 1, 0
opts.train.isDataAug = 0; % we include rotation, scale, translation and color casting in the file cnn_train_hopefast
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

switch opts.modelType
  case 'HOPE', net = cnn_cifar_init_hope(opts) ;
end



if exist(opts.imdbPath, 'file')
  load(opts.imdbPath) ;
  
  if (opts.dataClass == 0)
	dataMean = mean(imdb.images.data(:,:,:,1:50000), 4);
	imdb.images.data = bsxfun(@minus, imdb.images.data, dataMean);
  end
  
  if opts.whitenData
	z = reshape(imdb.images.data,[],60000) ;
	sigma = z * transpose(z) / size(z, 2);%sigma is a n-by-n matrix
	%%perform SVD
	[U,S,V] = svd(sigma);
	disp('Image processing using ZCAwhitening');
    epsilon = 0.01;
    z = U * diag(1./sqrt(diag(S) + epsilon)) * U' * z;
	imdb.images.data = reshape(z, 32, 32, 3, []) ;
  end
  
else
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
imdb.images.data = single(imdb.images.data);

[net, info] = cnn_train_hope(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

valError = min(info.val.error(1, :));
fprintf('Minimum validation error is %.6g \n', valError);

% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
%if rand > 0.5, im=fliplr(im) ; end

% --------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
