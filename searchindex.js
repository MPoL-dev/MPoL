Search.setIndex({docnames:["api","changelog","developer-documentation","index","installation","tutorials/PyTorch","tutorials/crossvalidation","tutorials/gridder","tutorials/optimization","units-and-conventions"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["api.rst","changelog.rst","developer-documentation.rst","index.rst","installation.rst","tutorials/PyTorch.ipynb","tutorials/crossvalidation.ipynb","tutorials/gridder.ipynb","tutorials/optimization.ipynb","units-and-conventions.rst"],objects:{"mpol.connectors":{GriddedResidualConnector:[0,1,1,""],index_vis:[0,4,1,""]},"mpol.connectors.GriddedResidualConnector":{forward:[0,2,1,""],ground_amp:[0,3,1,""],ground_mask:[0,3,1,""],ground_phase:[0,3,1,""],ground_residuals:[0,3,1,""],sky_cube:[0,3,1,""]},"mpol.coordinates":{GridCoords:[0,1,1,""]},"mpol.coordinates.GridCoords":{check_data_fit:[0,2,1,""]},"mpol.datasets":{Dartboard:[0,1,1,""],GriddedDataset:[0,1,1,""],KFoldCrossValidatorGridded:[0,1,1,""],UVDataset:[0,1,1,""]},"mpol.datasets.Dartboard":{build_grid_mask_from_cells:[0,2,1,""],get_nonzero_cell_indices:[0,2,1,""],get_polar_histogram:[0,2,1,""]},"mpol.datasets.GriddedDataset":{add_mask:[0,2,1,""],ground_mask:[0,3,1,""],to:[0,2,1,""]},"mpol.gridding":{Gridder:[0,1,1,""]},"mpol.gridding.Gridder":{get_dirty_beam:[0,2,1,""],get_dirty_beam_area:[0,2,1,""],get_dirty_image:[0,2,1,""],sky_vis_gridded:[0,3,1,""],to_pytorch_dataset:[0,2,1,""]},"mpol.images":{BaseCube:[0,1,1,""],FourierCube:[0,1,1,""],HannConvCube:[0,1,1,""],ImageCube:[0,1,1,""]},"mpol.images.BaseCube":{forward:[0,2,1,""]},"mpol.images.FourierCube":{forward:[0,2,1,""],ground_amp:[0,3,1,""],ground_phase:[0,3,1,""],ground_vis:[0,3,1,""]},"mpol.images.HannConvCube":{forward:[0,2,1,""]},"mpol.images.ImageCube":{forward:[0,2,1,""],sky_cube:[0,3,1,""],to_FITS:[0,2,1,""]},"mpol.losses":{PSD:[0,4,1,""],TV_channel:[0,4,1,""],TV_image:[0,4,1,""],UV_sparsity:[0,4,1,""],edge_clamp:[0,4,1,""],entropy:[0,4,1,""],nll:[0,4,1,""],nll_gridded:[0,4,1,""],sparsity:[0,4,1,""]},"mpol.precomposed":{SimpleNet:[0,1,1,""]},"mpol.precomposed.SimpleNet":{forward:[0,2,1,""]},"mpol.utils":{fftspace:[0,4,1,""],fourier_gaussian_klambda_arcsec:[0,4,1,""],fourier_gaussian_lambda_radians:[0,4,1,""],get_Jy_arcsec2:[0,4,1,""],get_max_spatial_freq:[0,4,1,""],get_maximum_cell_size:[0,4,1,""],ground_cube_to_packed_cube:[0,4,1,""],log_stretch:[0,4,1,""],loglinspace:[0,4,1,""],packed_cube_to_ground_cube:[0,4,1,""],packed_cube_to_sky_cube:[0,4,1,""],sky_cube_to_packed_cube:[0,4,1,""],sky_gaussian_arcsec:[0,4,1,""],sky_gaussian_radians:[0,4,1,""]},mpol:{connectors:[0,0,0,"-"],coordinates:[0,0,0,"-"],datasets:[0,0,0,"-"],gridding:[0,0,0,"-"],images:[0,0,0,"-"],losses:[0,0,0,"-"],precomposed:[0,0,0,"-"],utils:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","property","Python property"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:property","4":"py:function"},terms:{"0":[0,3,5,6,7,8,9],"000":[6,9],"0000e":8,"0025":7,"005":[0,7,8],"00j":8,"01":8,"03":[6,8],"0500":8,"0526e":8,"0527e":8,"0578e":8,"05j":8,"06":8,"0625":8,"06716516993585":6,"07j":8,"0832":8,"0837":8,"0839":8,"0841":8,"0842":8,"0874":8,"0875":8,"0880":8,"0881":8,"0883":8,"08j":8,"0915":8,"0917":8,"0921":8,"0923":8,"0925":8,"0955":8,"095552746835345":8,"0960":8,"0962":8,"0966":8,"0967":8,"0968":8,"0996":8,"0x7f63c8430640":8,"0x7f63c8484a30":8,"0x7f63c85e1d00":8,"0x7f63c8602640":8,"1":[0,2,5,6,7,8,9],"10":[0,3,5,6,7,8,9],"100":[2,5,6,9],"1000":[5,6],"1004":8,"1009":8,"1010":8,"1011":8,"1037":8,"1046":8,"1048":8,"1052":8,"1054":8,"1055":8,"11":[5,6,7,8,9],"113":9,"12":[0,5,6,7,8,9],"1250":8,"13":[5,6,7,8,9],"14":[5,6,7,8,9],"145":[6,8],"1481e":8,"15":[6,7,8],"16":[8,9],"167":9,"17":[8,9],"1791e":8,"1797e":8,"18":[8,9],"180":6,"1800e":8,"19":8,"1980e":8,"1d":0,"1e":[0,6],"1min":8,"1s":8,"1st":0,"2":[0,1,2,5,6,7,8,9],"20":[5,8],"2019":0,"2020":3,"202181115487862":6,"21":[5,8],"219":[6,8],"21j":8,"22":8,"22j":8,"23":8,"230":9,"230000000000":0,"24":[0,5,8],"25":5,"2500":8,"260":5,"2898e":8,"2d":[0,7,8],"2x3":5,"3":[0,4,5,6,7,8,9],"300":8,"33":9,"334":9,"340":9,"35":7,"3647603":3,"38":9,"384":9,"39":[6,8],"3blue1brown":5,"3d":[0,9],"3mm":9,"4":[0,3,5,6,7,8,9],"415210661003532":6,"42":6,"4498439":[6,7,8],"4581e":8,"47":5,"4745e":8,"480x480":6,"4885e":8,"4891e":8,"4895e":8,"4899e":8,"49":6,"4901e":8,"4902e":8,"4903e":8,"5":[5,6,7,8,9],"50":9,"500":[6,8,9],"512":0,"5264e":8,"5266e":8,"5281":3,"54":5,"567":9,"57":9,"6":[4,5,6,7,8,9],"600":6,"61":8,"6390e":8,"6480e":8,"650160":6,"7":[5,6,7,8,9],"7632e":8,"7633e":8,"767":9,"77":9,"8":[0,5,6,7,8,9],"800":[6,7,8],"8083e":8,"8154e":8,"8196e":8,"9":[5,6,7,8],"90":7,"9975":7,"9s":8,"boolean":0,"break":[2,3,6],"case":[0,2,5,8,9],"catch":1,"class":[0,2,9],"default":[0,8,9],"do":[0,2,4,5,6,7,8,9],"export":[0,6,8],"final":[0,5,6,8],"float":[0,5],"function":[0,1,2,3,6,9],"import":[2,3,4,5,6,8],"int":[0,6,9],"long":[0,2,5,9],"new":[0,2,5,6],"null":0,"return":[0,5,6],"short":2,"true":[0,5,6,7,8],"try":[6,8],"while":[2,5,6],A:[0,2,3,5,8,9],And:[0,2,6,7,8],As:[2,6,9],Be:2,But:[5,6,8],By:0,For:[0,2,3,5,6,7,8,9],If:[0,2,3,4,5,6,7,8,9],In:[0,2,5,6,8,9],It:[2,3,6,7,8,9],Its:0,No:2,Of:8,One:[2,5,6],The:[0,1,2,3,5],Then:[2,6,9],There:[0,2,3,5,6,7,9],These:[0,2,5,8],To:[0,2,3,5,6,7,8,9],With:[5,8],_:9,__init__:[6,8],__version__:4,_build:2,_execution_engin:[6,8],_static:8,a8:9,a_tensor:5,ab:5,abil:[1,5,6],abl:[2,5],about:[2,3,5,6,7,8],abov:[0,2,5,9],absolut:[5,7],accept:0,access:[0,4,7,8],accomplish:2,accord:0,account:[2,9],accur:[0,6],achiev:2,across:[0,9],act:0,action:2,activ:2,actual:6,ad:[1,2,5],adam:6,add:[2,5,8],add_mask:0,add_scalar:6,addit:[0,2,3,6,8,9],addition:2,adequ:6,advanc:8,advers:6,advic:2,affect:6,after:[0,2,5,9],against:[0,6,8],aggreg:6,aim:2,algorithm:[6,8],all:[0,1,2,5,6,7,8,9],allevi:6,allow:5,alma:[2,3,6,7,8,9],along:[0,9],alpha:[6,7],alreadi:[0,2,5,6,7,9],also:[0,2,6,7,8,9],alter:5,altern:2,alwai:[2,6],ambiti:8,among:0,amount:[0,8],amplitud:0,an:[0,2,3,5,6,7,9],an_arrai:5,analyt:[0,5],angl:[0,6,9],angu:5,ani:[0,2,3,6,8,9],annoi:9,anoth:5,another_arrai:5,another_tensor:5,answer:5,antenna:[6,9],anyth:2,apart:9,apertur:3,api:[3,7,8],appear:[0,5,7,9],append:[5,6,8],appendix:0,appli:[0,8,9],applic:[0,5,6,8],appreci:2,approach:6,appropri:[0,5,6,9],approv:2,approx:9,approxim:[0,6,9],approxm:0,ar:[0,2,4,5,6,7,8,9],arang:6,arbitrari:5,arcsec:[0,1,7,9],arcsecond:[0,9],arctan2:[6,9],area:[0,9],areial:9,aren:8,arg:0,argu:8,argument:[0,2,9],aris:9,around:[0,6],arrai:[0,3,5,6,7,8,9],arrang:0,arriv:0,art:3,ascend:[0,5],aspir:0,assert:1,assertionerror:0,assess:[2,6],associ:5,assum:[0,2,6,9],assumpt:[6,8],astronom:[7,9],astronomi:6,astrophys:6,astropi:[2,6,7,8],atacama:3,atan2:9,aten:[6,8],attach:[0,2,7,8],attribut:[0,5],augment:0,author:3,auto:3,autodifferenti:[5,9],autodifferentiaton:5,autoexecut:7,autograd:[5,6,8],autom:[2,5],automat:[0,2,5,8],avail:[0,2,8],averag:[0,6,7],avoid:2,ax:[6,7,8,9],axessubplot:6,axi:[0,9],azimuth:[0,6],b:[0,2],back:[2,8,9],background:0,backward:[5,6,8,9],bake:6,band:9,base:[0,5,8,9],base_cub:[0,8],basecub:[0,6,8],baselin:[0,9],basetemp:2,basi:8,basic:[0,8],bayesian:6,bcube:[0,8],bcube_numpi:8,bcube_pytorch:8,beam:[0,6,7,9],becaus:[0,6,7,8,9],becom:2,been:[0,2,6],befor:[5,6,7,8,9],begin:2,behind:2,being:[0,2,6],believ:8,below:[2,3],benefit:5,best:[5,6,8],better:[6,8],between:[0,6,7,8,9],beyond:9,bia:8,big:6,bin:[0,2],bit:8,bl:0,blank:8,blog:5,blow:5,bool:0,both:[0,9],bottom:[0,5,7],bound:8,boundari:0,bracewel:[0,9],branch:2,breviti:0,brianna:[1,3],brigg:[0,1,7,9],bright:[0,9],broadcast:0,browser:2,bug:[2,8],build:[1,3,6],build_grid_mask_from_cel:0,built:[2,3,5],bundl:[1,8],c1:5,c43:6,c:[5,6],cach:[6,7,8],cal:9,calcul:[0,3,6,8],calculu:5,call:[0,6,7,8,9],can:[0,2,4,5,6,7,8,9],cannot:[0,2,5],capabl:3,capac:7,carri:[0,2,8,9],cartesian:9,casa:[0,6,7,8],casa_convent:7,cast:[6,8],cd:[2,4],cell:[0,2,6,7,9],cell_index_list:0,cell_siz:[0,1,6,7,8,9],center:[0,5,6,9],central:[2,6,7,8],centroid:0,certain:9,chain:5,chan:[6,7,8],chang:[1,2,5,8],changelog:3,channel:[0,6,7,8,9],chapter:[7,9],characterist:8,chart:[1,2],check:[0,2,3,5,6,7,9],check_data_fit:0,checkout:2,chi:[0,6,8],choic:[0,5,6],choos:[1,3,5,9],chose:[5,8],chosen:5,chunk:[0,6],citat:1,cite:3,clean:[2,6,7,8],clear:9,cli:2,click:2,clone:4,close:[5,6],cluster:6,cmap:6,co:[0,6,7],coars:1,code:[0,3,8,9],codebas:[2,9],collabor:2,collaps:0,color:5,colorbar:8,column:9,com:[2,4],combin:6,come:[0,6],command:[2,6,7],commit:2,common:[5,8,9],commonli:[0,9],commun:[6,8],compact:0,compar:[0,2,5,6,7,8],comparison:8,complet:[2,5,6,8],complex128:8,complex:[0,5,6,7,8],compli:2,complic:9,compon:[0,7,8,9],compos:0,comprehens:2,comput:[0,5,6,8],concept:[5,6,8],concern:6,confid:6,config:[2,6],configur:[2,6],confirm:7,conflict:2,confus:9,conjug:[0,6,7,8],connect:[0,8],connector:[3,6],conserv:8,consid:[0,5,6,9],consider:9,consist:5,constant:[0,8],constel:3,constraint:8,construct:[0,8],consult:[6,9],contain:[0,2,5,6,7,8],content:2,context:[6,9],continu:[3,5,6,8],continuum:[0,3,7],contribut:[3,4],control:2,conv_lay:8,conveni:[0,7,9],convent:[0,2,3,6,7,8],converg:8,convers:[0,8],convert:[0,2,6,7,8,9],convolut:0,coord:[0,6,7,8],coordin:[3,5,6,7,8,9],copi:[0,2,6,8],core:[0,8],correct:[2,9],correctli:[2,8,9],correl:0,correspond:[0,2,6,7,8,9],cosin:9,could:[0,2,5,8],count:[0,8,9],cours:[2,6,8],cov:2,cover:[2,7,8],coverag:6,cpp:[6,8],cpu:8,creat:[0,2,5,6,7,8,9],criterion:5,cross:[1,3],cross_valid:6,cube:[0,1,6,7,8],current:[0,2,4,5,8,9],cv:[0,6],cycl:9,czekala:3,d:[3,4,5,6,7,8,9],d_:0,danger:5,daniel:[7,9],dark:5,dartboard:[0,1,6],data:[0,2,3,5,6,8,9],data_im:[0,6,7,8],data_r:[0,6,7,8],data_vi:0,dataconnector:0,datapoint:0,dataset:[1,2,3,5,6,7],datasetconnector:1,datasetgrid:0,date:2,deal:9,debug:0,dec:[0,9],decid:0,decreas:9,deep:8,def:[5,6],defacto:1,defalt:0,defin:[0,5,6,7,8,9],definit:0,degre:0,delet:2,deliv:[0,7],delta:[6,7,9],delta_i:0,delta_l:0,delta_m:0,delta_x:0,demonstr:[6,8,9],dens:7,densiti:[0,1,6],depend:[6,7],deriv:[0,5],descent:3,describ:[2,6,8,9],descript:0,design:[0,6,8],desir:[0,8],detach:[6,8],detail:0,determin:[4,5],dev:[2,3,4],develop:[3,4],devic:0,dft:9,diagnost:[3,8,9],diagram:8,dict:0,dictionari:6,did:6,diff:2,differ:[2,6,7,8,9],differenti:[3,5],dimens:[0,6,7,8],dimension:[0,7,8,9],direct:[0,2,5,8,9],directli:[0,7],directori:2,dirti:[0,6,7,8,9],dirty_imag:1,discard:[6,8],discret:3,discuss:[2,3,8],displai:[8,9],dissimilar:8,distribut:[6,9],diverg:[5,8],dl:0,dm:0,doc:[1,2],docstr:2,document:[0,3,4,5,7,8],doe:[6,9],doesn:[2,6,8],doi:3,domain:[0,9],don:[0,2,6,7,8,9],done:[0,2,6,8],doubl:0,down:[3,5,7,9],download:[2,4,7],download_fil:[6,7,8],downward:5,draft:2,draw:[2,6],drawn:[6,9],dset:[6,8],dtype:8,du:0,dual:0,due:[2,6],durat:6,dure:0,dv:0,dx:5,dy:5,e:[0,2,7,9],each:[0,2,5,6,7,9],easi:[0,5,8],easier:2,easili:[5,8,9],east:[0,7,9],edg:0,edge_clamp:0,edit:2,effect:[0,9],effici:5,eht:[0,1],either:[0,7],element:[0,9],ell:0,emiss:6,emphasi:2,enclos:0,encourag:0,end:[0,6],enforc:[0,1,8],enorm:5,enough:[6,8],ensur:[0,2],entropi:[0,1],enumer:6,environ:3,envis:2,epoch:6,epsilon:0,eqn:9,equal:[0,5,6],equat:[0,5,9],equival:[0,6,9],error:[2,8],especi:[2,6],essenti:[0,8],estim:[0,8],etc:8,evalu:[0,5,6,8],even:[0,2,6,8,9],everi:2,evid:6,examin:8,exampl:[0,2,5,6,8,9],excel:2,excit:8,execut:2,exercis:5,exist:[0,2],exoplanet:2,exp:[0,9],expect:[0,5,6,9],experi:8,explain:2,explicitli:9,explor:6,express:9,extend:3,extens:2,extent:[0,6,7,8],extra:[0,2],extra_requir:2,extrem:8,f:[0,5,9],f_0:0,f_1:0,f_2:0,f_3:0,f_:[0,6],face:9,fact:[8,9],factor:0,fail:2,fals:[0,6,7],familiar:[2,6,7],far:[5,8],fast:9,favor:1,favorit:2,fcube:0,fear:2,featur:2,feb:3,fed:0,feed:[0,7,8],feedback:2,feel:[5,8],fell:0,felt:5,fetch:2,few:[6,8],fft2:9,fft:0,fftshift:[0,9],fftspace:0,fidel:6,field:9,fig:[6,7,8],figsiz:[6,7],figur:[2,6,9],file:[0,2,6,7,8],filenam:[7,8],find:[2,5,6],fine:7,finish:2,first:[0,2,5,6,7,8,9],fit:[0,6,8,9],fix:[0,2],flat:5,flatten:6,flayer:6,flexibl:3,flip:[0,9],float64:[0,8],flowchart:1,fname:[0,6,7,8],focu:3,fold:[0,3],folder:2,follow:[0,1,2,5,6,7,8,9],fool:8,form:[0,2,5,6,8],format:[0,2,6,8],forward:[0,6,8,9],found:2,fourier:[0,3,6,7],fourier_gaussian_klambda_arcsec:0,fourier_gaussian_lambda_radian:0,fouriercub:[0,6],frac:[0,5,9],framework:[3,5],free:[2,8],frequenc:[0,1,6,7,8,9],frequent:7,from:[0,3,5,6,7,8,9],full:[0,2,6,8],fulli:[0,6,9],functionil:0,fundament:8,further:8,futur:8,g:[0,2,3,9],gap:6,gaussian:0,gener:[0,2,6,8],geometr:9,get:[3,5,6,7,8],get_dirty_beam:0,get_dirty_beam_area:0,get_dirty_imag:[0,1,6,7,9],get_jy_arcsec2:0,get_max_spatial_freq:0,get_maximum_cell_s:0,get_nonzero_cell_indic:0,get_polar_histogram:0,ghz:[7,9],git:[2,4],github:[1,3,4,8],give:2,given:[0,5],gnbu:6,gnu:2,go:[0,2,5,9],goal:[0,2,6,8],good:[5,7],grad:[5,8],grad_fn:8,gradient:[0,3,6],graph:[5,8],greater:0,greatli:0,grid:[3,6,9],grid_vis:0,gridcoord:[0,1,3,6,8],gridded_im:7,gridded_r:7,gridded_vi:7,griddeddataset:[0,6],griddedresidualconnector:[0,6],gridder:[0,1,2,3,6,8,9],ground:[0,5,9],ground_amp:0,ground_cub:0,ground_cube_to_packed_cub:[0,1,9],ground_mask:[0,6],ground_phas:0,ground_residu:0,ground_vi:0,gt:[6,8],guid:2,guidanc:2,guidelin:2,ha:[0,2,7,8,9],had:[6,9],hand:5,handi:6,hann:[0,1],hannconvcub:[0,1],happen:8,hard:6,hast:6,have:[0,2,3,4,5,6,7,8,9],haven:5,header:[0,2],header_kwarg:0,help:[2,3,6,8],henc:2,here:[2,5,6,7,8,9],hermitian:[0,6],high:6,highlight:5,hill:5,histogram:0,hogg:6,hold:0,hole:6,home:2,hook:2,hopefulli:8,host:[2,3,4,6],hostedtoolcach:[6,8],hour:6,how:[0,2,3,5,6,7,8,9],howev:[2,5,6,7,8,9],hspace:6,html:2,htmlcov:2,http:[2,3,4,6,7,8],hundr:8,hyperparamet:6,hz:0,i:[0,2,6,7,8,9],i_:[0,9],i_i:0,ian:3,iancz:3,icub:[0,6,8],idea:[0,2,6,7,8],ideal:6,ident:[0,9],identifi:6,ifft:9,ifftshift:9,ignor:6,ill:[8,9],im:[0,8],imag:[1,3,5,6],imagecub:[0,6,8],imaginari:[0,6,7,8],img:[6,7],img_ext:[0,6,7,8],implement:[0,1,2,6,7,9],implicitli:[0,6],implment:9,improv:[2,6,8],imshow:[0,6,7,8,9],inclin:5,includ:[0,1,2,5,6,9],incorpor:[1,2,6],increas:[0,9],ind:6,independ:5,index:[0,2,3,9],index_vi:0,indic:[0,9],individu:[0,2,6,8],infer:6,inform:[0,2,3,7,8,9],infti:9,inher:6,inherit:0,initi:[0,2,5,6,8],input:[0,7,9],instal:[1,3,5,7],instanc:0,instanti:[0,7],instead:[1,5,6,8],instruct:2,int_:9,integ:0,integr:[2,9],intens:[0,9],intent:2,interest:[2,4,6,7],interfac:2,interferomet:[3,9],interferometr:[0,6,9],interferometri:3,intermedi:[6,9],intern:[0,6,8,9],interpol:[6,7,8],interpret:0,introduc:[1,2],introduct:3,introductori:8,invalid:[0,5],invers:[8,9],invert_xaxi:6,investig:2,invoc:2,invok:2,involv:0,io:1,ipynb:2,ipython:8,isn:[2,8],issu:[0,2,4,8],item:[5,6,8],iter:[0,3,5,6,7],its:[0,2,5,9],itself:[0,2,6,8],iv:[0,1],j:[6,9],janski:3,jax:5,jean:0,job:6,join:3,js:[1,2],jupyt:[1,2],jupytext:[1,2],just:[0,2,6,7,8,9],jy:[0,1,7,9],k:[0,3,9],k_fold:6,k_fold_dataset:6,karl:3,keep:[0,2,5,6,8,9],kei:[2,5],kept:5,kernel:[0,2,3],keystrok:0,keyword:0,kfoldcrossvalidatorgrid:[0,1,6],kilolambda:0,klambda:[0,7],km:9,know:[2,5,8,9],kw:[6,7],kwarg:0,l:[0,9],l_1:0,l_2:0,laid:0,lambda:[0,6,9],lambda_:6,lambda_spars:6,lambda_tv:6,land:0,laplac:5,larg:[3,5,6,8],large_step_s:5,larger:[0,5],last:[0,5],lastli:8,later:[5,8],latest:4,layer:0,layout:0,lead:0,learn:[2,5,6,8],least:[0,6],leav:9,left:[0,5,6,7,9],leftrightharpoon:0,len:6,length:[0,9],lens:5,let:[0,5,6,7,8,9],level:6,lib:[6,8],lieu:0,lifecycl:2,like:[0,1,2,3,4,5,6,7,8,9],likelihood:[0,3,5,8],limit:[0,2,6],line:[2,3,5,7],linear:0,linearli:0,linewidth:6,link:[2,9],linspac:5,list:[0,2,5,6],literatur:8,ll:[0,5,6,7,8],lmv:0,ln:0,load:[6,7,8],local:2,locat:[0,2,5,6],lock:7,log:[0,2],log_stretch:0,logarithm:[0,1],loglinspac:0,logo:[6,7,8],logo_cub:[6,7,8],logspac:0,longer:[0,5],look:[0,2,3,5,6,7,8,9],loomi:[1,3],loop:[3,5],loos:[0,7],loss:[1,3,6],loss_track:8,lot:2,love:2,low:6,lower:[0,6,7,8,9],lr:[6,8],lt:[6,8],ly:6,m:[0,2,8,9],m_:0,m_linear:0,machin:[2,6,8],made:[1,2,7],magic:5,mai:[0,2],main:[0,2],mainli:[0,7],make:[2,5,6,7,8,9],makefil:2,mani:[0,2,5,6,8,9],manner:[0,8],map:[0,6],mask:[0,3,6],match:0,materi:2,mathemat:[5,9],mathrm:[0,1,6,9],mathtt:9,matmul:5,matplotilb:9,matplotlib:[0,5,6,7,8,9],matric:5,matrix:5,max:0,max_freq:0,max_grid:0,maxim:8,maximum:[0,3,5,7,8],mcmc:6,mean:[0,2,5,6,8,9],meant:7,measur:[0,6,7,9],mechanist:5,merg:2,mermaid:[1,2],meta:8,metadata:2,meter:9,method:[0,5,7,9],metric:6,metropoli:6,middl:0,might:[0,2,5,6,8,9],migrat:1,millimet:3,minim:[0,8],minimum:[0,5,8],mirror:7,miss:[5,6],mix:0,mkdir:2,mmd:8,mmdc:2,mock:[2,6,7,8],mode:6,model:[0,3,6,9],model_vi:0,modifi:2,modul:[1,3,8],moment:5,month:3,more:[0,2,3,5,6,7,8,9],most:[0,5,6,7,8,9],mostli:6,move:[0,1,7],mpol:[0,1,4,5,6,7,8,9],mpoldataset:2,ms:7,much:8,mulbackward0:8,multi:[0,2,6,8],multipl:[0,5,9],multipli:5,multivari:5,must:[0,9],my_feature_branch:2,mycoord:0,mymeasurementset:7,n:0,n_log:0,n_v:0,nabla:5,name:[0,2],narrow:8,nativ:[0,1,6,8],natur:[0,1,7],nbsphinx:2,nchan:[0,6,7,8,9],ncol:[6,7],nearli:[0,5,6,7,9],neat:2,necessari:[2,8],need:[0,2,5,6,7,8,9],neg:6,network:[0,8],neural:8,never:0,newer:2,next:[0,6],nll:[0,6],nll_grid:[0,1,6,8],nn:0,node:0,nois:7,non:[0,6,9],none:[0,6,7,8],norm:[0,9],normal:[0,2,6,7,8,9],north:[0,7,9],note:[0,2,5,7],notebook:[1,2],noth:5,notic:2,now:[1,5,7,8],np:[0,5,6,7,8],npix:[0,6,7,8,9],npseed:[0,6],npz:[6,7,8],nrow:[6,8],ntheta:0,nu:0,number:[0,6,7,9],numer:[7,9],numpi:[0,5,6,7,8,9],nvi:[0,6,7],nyquist:0,object:[0,1,3,6],observ:[3,6,7,9],obtain:5,occasion:9,off:8,offend:2,offici:2,offset:0,often:[6,8],ok:9,omega:0,onc:[0,2,6],one:[0,2,5,6,7,8],ones:8,onli:[0,2,5,6,8,9],open:[0,2],oper:[0,5,9],opportun:9,oppos:0,opt:[0,6,8],optim:[0,1,3,6],option:0,orang:5,order:[0,8,9],ordereddict:8,orderli:2,org:[3,6,7,8],organ:[2,3,8],orient:[8,9],origin:[0,2,6,7,8,9],other:[2,3,5,6],otherwis:[0,2,5,7],our:[2,3,5,6,8],ourselv:[7,8],out:[0,2,3,6,8,9],outdat:2,outlin:8,output:[0,1,2,3,5,9],over:[0,5,9],overlap:0,overwrit:0,own:[0,2,6,7],p:0,p_i:0,pack:0,packag:[0,1,2,3,4,5,6,8,9],packed_cub:0,packed_cube_to_ground_cub:[0,1,9],packed_cube_to_sky_cub:[0,1,8,9],page:[0,1,2,3,9],pair:[0,2,6],pandoc:2,paperspac:5,par:6,paramet:[0,6,7,8,9],parameter:[0,9],parlanc:8,part:[0,2,5,6,8],particular:4,partit:[0,6],pass:[0,2,5],passthrough:0,pasthrough:0,pattern:[2,6],peak:0,penalti:0,per:[0,9],perform:[0,5,6],perhap:2,permit:0,permut:0,perspect:9,ph:[7,9],phase:[0,3],phi:[0,6,9],phi_cel:0,phi_edg:0,phrase:0,pi:[0,9],pick:5,piecewis:0,pip:[1,2,3],pitfal:9,pixel:[0,1,6,8],pixel_map:0,pkgname:[6,7,8],place:[2,5,6],plane:[0,6,7,9],pleas:[0,2,3,4,6,8],plot:[0,5,6,8,9],plotsdir:2,plt:[5,6,7,8],plug:5,plugin:2,png:2,point:[0,5,6,8,9],polar:[0,9],popul:5,popular:8,posit:[0,1,5,8,9],possibl:[0,2,6,8],post:[2,5],posterior:6,potenti:[3,6],power:[0,1,6,8],practic:9,pre:[0,2],preced:2,precis:7,precompos:[1,3,6,8],predict:[6,8],prefactor:[0,6,9],preliminari:1,premultipli:0,prepack:0,prepar:2,preprocess:2,preserv:[0,2],pretti:5,preview:2,previou:[5,6,8],previous:0,prime:[0,6,7],primit:8,principl:2,print:[4,5,6,7,8],prior:[0,5],prior_intens:0,probabilist:0,probabl:8,probe:[2,6],problem:[3,8],procedur:9,proceed:5,process:[0,1,2,5,6,8,9],prod_arrai:5,prod_tensor:5,produc:[0,2,6,7,8,9],product:[0,2,5,7,8,9],program:[2,5],progress:2,project:2,prolat:1,promot:0,propag:8,proper:5,properli:[2,9],properti:[0,7],propos:9,provid:[0,2,3,5,6,8,9],psd:0,psf:[0,7],publicli:2,publish:3,pull:[0,2,4,8],purpos:[0,6,7,8,9],push:2,put:6,py:[2,6,8],pypi:[2,4],pyplot:[0,5,6,7,8,9],pytest:2,python3:[2,6,8],python:[3,4,6,7,8],pytorch:[0,3,6,9],q:[0,6,9],q_:0,q_cell:0,q_edg:0,q_max:0,qs:0,quantiti:[0,9],queri:2,question:[3,8],quick:[0,2,9],quickli:[5,6],quickstart:7,quit:[6,7,9],r:[0,5,6,7,8,9],ra:0,radial:[0,6],radian:[0,9],radio:7,rai:[6,9],rais:[0,2,4,8],random:[0,6],randomli:6,rang:[0,2,6,8,9],raster:6,rate:[5,8,9],rather:8,rayleigh:0,re:[0,2,4,5,6,7,8,9],reach:[2,5,6,8],read:[1,2,7],readi:2,real:[0,6,7,8,9],realist:8,realli:6,reason:[0,5,8,9],rebuild:2,recap:3,recogniz:9,recommend:[0,2,6],recommit:2,reconstruct:3,record:[6,7,8],recov:6,redo:2,refer:[0,2,9],reflect:[2,8],region:0,regular:[0,3,5,6,7,8],rel:[0,1,5],relat:9,relationship:[0,9],releas:3,relev:8,reli:2,reliabl:2,remain:6,rememb:[2,7],reminisc:0,remot:2,remov:[1,8],renam:2,render:[2,5],repeat:[2,5],replac:6,replic:6,replot:5,repo:2,report:2,repositori:[2,3,4],repres:[0,8,9],represent:[0,3,7,8],reproduc:0,request:[0,2,8],requir:[0,2,4,8,9],requires_grad:[0,5],research:3,residu:0,resolut:7,resourc:[2,3],respect:[0,5,8,9],rest:8,restrict:0,result:[0,3,5],retain:5,review:2,revisit:6,rfft:0,right:[0,2,5,7,9],rigor:2,rml:[0,3,5,6,8,9],robust:[0,1,7],role:8,root:[0,2],rotat:[0,6],rough:0,routin:[0,1,7,8,9],row:9,rst:2,rtest:6,rtrain:6,rule:5,run_backward:[6,8],ryan:[1,3],s:[0,2,5,6,7,8,9],sai:[5,9],same:[0,5,6,7,8,9],sampl:[0,6,9],sampler:6,satisfi:2,save:[0,2,7],saw:8,scale:[0,5],scatter:[5,6],scene:2,scheme:[6,7],scope:2,score:6,scratch:6,script:[0,2],second:5,see:[0,2,4,5,6,7,8,9],seed:0,seen:9,self:[0,2],semin:9,sens:[3,6],separ:[6,7],sequenc:9,serv:9,session:2,set:[0,3,5,6,7,8,9],set_ticklabel:6,set_titl:[6,7],set_xlabel:[6,7,8],set_ylabel:[6,7,8],setup:[2,3,8],sever:[2,5,6,7,8,9],sgd:8,shape:[0,6,7,8,9],shell:2,shift:[0,9],ship:2,shot:7,should:[0,2,6,7,8],show:[5,6,8],show_progress:[6,7,8],shown:5,side:[0,5],sigma:0,sigma_i:0,sigma_l:0,sigma_m:0,sigma_x:0,signific:[0,2],significantli:6,similar:[0,5,6,7,8],simpl:[0,8,9],simplenet:[0,1,6,8],simplest:8,simpli:0,simplic:0,simplifi:0,simul:[6,7],simultan:0,sin:0,sinc:[0,2,5,8,9],singl:[0,5,6,7,8,9],single_channel_estim:0,site:[6,8],situat:9,size:[0,5,6,7],sizeabl:6,sky:[0,6,8,9],sky_cub:[0,6,8],sky_cube_to_packed_cub:[0,1,9],sky_gaussian_arcsec:0,sky_gaussian_radian:0,sky_model:0,sky_vis_grid:0,slice:[0,6],slightli:6,slope:5,slope_curr:5,slope_large_step_curr:5,slope_start:5,small:[2,5,6,8,9],smooth:0,so:[2,5,6,8,9],soften:0,softplu:0,softwar:[2,3],solut:2,solv:5,some:[0,2,5,6,8,9],someon:2,someth:[0,9],sometim:6,somewhat:6,somewher:5,sophist:6,sourc:[0,2,3,8,9],south:[0,7],space:[0,6,7,8,9],span:0,spars:[0,3,6],sparsiti:[0,6],spatial:[0,1,6,7,8,9],special:2,specif:[0,2],specifi:[0,2,7,9],spectral:[0,1,3,7],speed:[0,5],spheroid:1,spheroidal_grid:1,sphinx:2,split:[0,6],spot:8,spread:0,sqrt:[0,6,9],squar:[0,5],squeez:[6,8],src:[6,8],stabl:[2,4],stale:2,stand:5,standard:[6,9],start:[0,2,3,5,6,7,8],state:[0,3,8],state_dict:8,statement:1,steepest:5,step:[0,2,5,6,8,9],step_siz:5,steradian:0,still:[0,2,5,6,7,8],stochast:8,stop:[2,5],store:[0,1,2,5,6,8],str:0,strang:5,strength:[0,6],stretch:0,strictli:2,string:0,strive:2,strong:[6,8],structur:6,submillimet:3,submit:[2,8],subpartit:0,subplot:[6,7,8],subplots_adjust:[6,7],subscript:9,subsect:2,subselect:6,subtleti:9,success:2,successfulli:2,suffici:8,suggest:[2,6],suit:[2,8],sum:[0,5,6],sum_:[0,9],sum_i:0,summarywrit:6,superset:2,suppli:0,support:[0,3],sure:[2,5,6],svg:8,swap:[7,8],swap_convent:7,sweep:6,sy:8,symmetr:0,sync:2,synthes:[8,9],system:[2,4,9],t:[0,2,5,6,7,8,9],t_b:0,tabl:2,tag:2,take:[0,2,3,5,7,8,9],tangent:5,taper:0,taper_funct:0,tclean:[0,7],technic:0,techniqu:6,tediou:2,tell:6,temperatur:0,temporari:2,tensor:[0,3,6,8],tensorboard:6,term:[0,5,6,8,9],test:[0,3,6],test_chan:6,test_dset:6,test_mask:6,test_scor:6,text:[5,6,8],th:[0,6],than:[0,2,5,6,9],thank:[2,9],thankfulli:9,thei:[0,2,5,6,8,9],them:[0,2,6],themselv:0,theorem:0,therefor:[0,9],thermal:[0,7],thesi:[7,9],thi:[0,1,2,5,6,7,8,9],thing:[2,5,8],think:[0,2,8],those:[0,2,6,7,8],though:[0,2,8],thought:[6,9],three:[0,7,9],through:[0,2,6,8],throughout:[2,9],thu:0,time:[2,5,6,8,9],titl:3,tm:[6,7,8,9],to_fit:0,to_pytorch_dataset:[0,6,8],togeth:[0,5,6,8],told:2,too:[1,2,5,6,8],tool:[2,6],top:[0,3,7],topic:2,torch:[0,5,6,8,9],total:[0,6,8],toward:[5,8],track:[5,8],train:[0,3,6],train_and_imag:6,train_chan:6,train_dset:6,train_mask:6,trainabl:0,transform:[0,3],translat:[0,9],trial:8,trigger:[6,8],troubl:[2,4],truth:8,tsm:7,tsv:6,tune:[0,6,8],tupl:0,turn:[0,6],tutori:[1,5,6,7,8],tv:0,tv_channel:0,tv_imag:[0,6],tweak:6,twice:8,two:[0,2,7],type:[0,6,9],typic:[6,9],typo:2,u:[0,6,9],u_bin_max:0,u_bin_min:0,u_cent:0,u_edg:0,uk:6,ul:9,uncertainti:8,undefin:0,under:2,understand:5,unfamiliar:2,ungrid:0,uniform:[0,7,8],uniqu:7,unit:[0,1,3,7],uniti:0,unless:0,unnorm:6,unpack:[0,9],unprim:0,unsampl:8,until:[2,5],untouch:9,up:[0,3,5,6,8,9],updat:[1,2,4,5,6,8],upgrad:3,upload:2,upon:2,upsid:7,upstream:2,url:3,us:[0,1,2,3,5,6,7,8,9],useabl:2,user:[0,1,2,8,9],userwarn:[6,8],usual:[6,7,9],util:[2,3,6,7,8,9],uu:[0,6,7,8],uu_vv_point:0,uv:[0,6],uv_spars:0,uvdataset:0,v0:[2,3],v:[0,6,9],v_:9,v_bin_max:0,v_bin_min:0,v_center:0,v_edg:0,va:5,val:6,valid:[1,3],vallei:5,valu:[0,1,2,5,6,8,9],valuabl:2,vanilla:2,vari:6,variabl:[0,2,5,6,8],variat:0,ve:[2,5,6,8],vector:0,veloc:0,venv:2,veri:[2,3,5,6,7],verifi:2,version:[2,3,4,9],vi:[0,6,7,8],via:[0,2,4,6,7,9],video:5,view:[0,9],villar:6,violat:0,vis_ext:[0,6],vis_grid:0,vis_index:0,visibilit:8,visibl:[0,6,7,8,9],visread:7,visual:[0,3,6,9],vk:6,vla:[3,9],vm:9,vv:[0,6,7,8],w:[0,8],w_i:0,wa:[0,9],wai:[2,5,6,9],walk:8,wall:8,want:[0,2,5,6,7,8,9],warrant:9,wavefunct:1,wavelength:9,we:[0,2,3,5,6,7,8,9],web:2,websit:3,wedg:0,weight:[0,1,6,7,8],weight_grid:0,weight_index:0,weight_indexd:0,well:[2,6],went:2,were:[5,6,7,8],west:[0,7],what:[0,2,5,6,8],whatev:2,when:[0,2,5,6,9],where:[0,5,6,9],whether:[0,2,6,7],which:[0,1,2,3,5,6,7,8,9],whole:6,whose:2,why:[2,5,6],width:0,william:5,window:[0,2],witheld:6,within:[0,2,5,9],without:[2,5,6,7,8],won:[6,7,8],work:[0,2,3,5,6,7,8],workflow:[2,6,8],world:9,worri:[2,6],worthi:2,worthwhil:8,would:[0,2,5,6,8,9],wouldn:[5,6],wrap:[6,8],wrapper:0,wrapup:3,write:[0,2,6],writer:6,written:2,wrong:2,wspace:[6,7],x64:[6,8],x:[0,5,9],x_:5,x_coord:5,x_current:5,x_input:5,x_large_coord:5,x_large_step:5,x_large_step_curr:5,x_large_step_new:5,x_new:5,x_start:5,xaxi:6,xlabel:5,xlim:5,xmax:5,xmin:5,y:[0,5],y_coord:5,y_current:5,y_large_coord:5,y_large_step_curr:5,y_large_step_new:5,y_new:5,y_start:5,yaml:2,yaxi:6,ye:5,year:3,yet:8,yield:8,ylabel:5,ylim:5,ymax:5,ymin:5,you:[0,2,3,4,5,6,7,8,9],your:[0,3,4,5,6,7,8],yourusernam:2,zawadzki:[1,3],zenodo:[2,3,6,7,8],zero:[0,5,6,7,8],zero_grad:[6,8],zeroth:9,zeta:0,zorder:5},titles:["API","Changelog","Developer Documentation","Million Points of Light (MPoL)","Installation","Introduction to PyTorch Tensors and Gradient Descent","Cross validation","Gridding and diagnostic images","Optimization Loop","Units and Conventions"],titleterms:{"0":1,"1":1,"4":1,"5":1,"break":8,"function":[5,8],"import":7,The:[6,7,8,9],addit:5,an:8,angular:9,api:0,best:2,build:[2,8],cach:2,calcul:5,changelog:1,choos:6,clone:2,code:2,connector:0,continu:9,contribut:2,convent:9,coordin:0,coverag:2,cross:[0,6],cube:9,data:7,dataset:[0,8],debug:2,depend:2,descent:[5,8],develop:2,diagnost:7,dimens:9,discret:9,document:2,down:8,environ:2,fft:9,flux:9,fold:6,fork:2,fourier:9,from:[2,4],github:2,gradient:[5,8],grid:[0,7,8],gridcoord:7,gridder:7,guid:3,imag:[0,7,8,9],instal:[2,4],interferometri:9,introduct:5,iter:8,k:6,light:3,loop:[6,8],loss:[0,8],million:3,model:[2,8],modul:0,mpol:[2,3],object:7,optim:[5,8],output:8,pack:9,pip:4,pixel:9,plot:2,point:3,practic:2,precompos:0,python:2,pytorch:[5,8],recap:8,releas:2,represent:9,resourc:5,result:6,run:2,set:2,setup:6,sourc:4,tensor:5,test:2,train:8,transform:9,tutori:[2,3],unit:9,up:2,upgrad:4,us:4,user:3,util:0,v0:1,valid:[0,6],view:2,virtual:2,visual:[7,8],wrapup:8,your:2}})