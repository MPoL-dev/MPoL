Search.setIndex({docnames:["api","changelog","developer-documentation","index","installation","tutorials/PyTorch","tutorials/crossvalidation","tutorials/gpu_setup","tutorials/gridder","tutorials/initializedirtyimage","tutorials/optimization","units-and-conventions"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,nbsphinx:3,sphinx:56},filenames:["api.rst","changelog.rst","developer-documentation.rst","index.rst","installation.rst","tutorials/PyTorch.ipynb","tutorials/crossvalidation.ipynb","tutorials/gpu_setup.rst","tutorials/gridder.ipynb","tutorials/initializedirtyimage.ipynb","tutorials/optimization.ipynb","units-and-conventions.rst"],objects:{"mpol.connectors":{GriddedResidualConnector:[0,1,1,""],index_vis:[0,4,1,""]},"mpol.connectors.GriddedResidualConnector":{forward:[0,2,1,""],ground_amp:[0,3,1,""],ground_mask:[0,3,1,""],ground_phase:[0,3,1,""],ground_residuals:[0,3,1,""],sky_cube:[0,3,1,""]},"mpol.coordinates":{GridCoords:[0,1,1,""]},"mpol.coordinates.GridCoords":{check_data_fit:[0,2,1,""]},"mpol.datasets":{Dartboard:[0,1,1,""],GriddedDataset:[0,1,1,""],KFoldCrossValidatorGridded:[0,1,1,""],UVDataset:[0,1,1,""]},"mpol.datasets.Dartboard":{build_grid_mask_from_cells:[0,2,1,""],get_nonzero_cell_indices:[0,2,1,""],get_polar_histogram:[0,2,1,""]},"mpol.datasets.GriddedDataset":{add_mask:[0,2,1,""],ground_mask:[0,3,1,""],to:[0,2,1,""]},"mpol.gridding":{Gridder:[0,1,1,""]},"mpol.gridding.Gridder":{get_dirty_beam_area:[0,2,1,""],get_dirty_image:[0,2,1,""],ground_cube:[0,3,1,""],to_pytorch_dataset:[0,2,1,""]},"mpol.images":{BaseCube:[0,1,1,""],FourierCube:[0,1,1,""],HannConvCube:[0,1,1,""],ImageCube:[0,1,1,""]},"mpol.images.BaseCube":{forward:[0,2,1,""]},"mpol.images.FourierCube":{forward:[0,2,1,""],ground_amp:[0,3,1,""],ground_phase:[0,3,1,""],ground_vis:[0,3,1,""]},"mpol.images.HannConvCube":{forward:[0,2,1,""]},"mpol.images.ImageCube":{forward:[0,2,1,""],sky_cube:[0,3,1,""],to_FITS:[0,2,1,""]},"mpol.losses":{PSD:[0,4,1,""],TV_channel:[0,4,1,""],TV_image:[0,4,1,""],UV_sparsity:[0,4,1,""],edge_clamp:[0,4,1,""],entropy:[0,4,1,""],nll:[0,4,1,""],nll_gridded:[0,4,1,""],sparsity:[0,4,1,""]},"mpol.precomposed":{SimpleNet:[0,1,1,""]},"mpol.precomposed.SimpleNet":{forward:[0,2,1,""]},"mpol.utils":{fftspace:[0,4,1,""],fourier_gaussian_klambda_arcsec:[0,4,1,""],fourier_gaussian_lambda_radians:[0,4,1,""],get_Jy_arcsec2:[0,4,1,""],get_max_spatial_freq:[0,4,1,""],get_maximum_cell_size:[0,4,1,""],ground_cube_to_packed_cube:[0,4,1,""],log_stretch:[0,4,1,""],loglinspace:[0,4,1,""],packed_cube_to_ground_cube:[0,4,1,""],packed_cube_to_sky_cube:[0,4,1,""],sky_cube_to_packed_cube:[0,4,1,""],sky_gaussian_arcsec:[0,4,1,""],sky_gaussian_radians:[0,4,1,""]},mpol:{connectors:[0,0,0,"-"],coordinates:[0,0,0,"-"],datasets:[0,0,0,"-"],gridding:[0,0,0,"-"],images:[0,0,0,"-"],losses:[0,0,0,"-"],precomposed:[0,0,0,"-"],utils:[0,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","property","Python property"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:property","4":"py:function"},terms:{"0":[0,3,5,6,7,8,9,10,11],"000":[6,11],"0000e":10,"0025":8,"005":[0,8,10],"00j":10,"01":10,"0225":9,"0234":9,"03":[6,7,9,10],"0301":9,"0500":[9,10],"0526e":10,"0527e":10,"0530":9,"0578e":10,"05j":10,"06":10,"0625":[9,10],"07j":10,"08j":10,"0x7f2a858db190":9,"0x7f2a859b4280":9,"0x7f2a86ba8eb0":9,"0x7f2dcc8541f0":10,"0x7f2dce914a90":10,"0x7f2dd0a95340":10,"0x7f2dd0acc2b0":10,"1":[0,2,5,6,7,8,9,10,11],"10":[0,3,5,6,7,8,9,10,11],"100":[2,5,6,11],"1000":[5,6],"100x":7,"1080":7,"11":[5,6,8,9,10,11],"113":11,"1161":10,"12":[0,5,6,8,9,10,11],"120":0,"1222":10,"124348303800103":10,"1250":[9,10],"1257":10,"1282":10,"13":[5,6,8,9,10,11],"1318":10,"1340":10,"1354":10,"1378":10,"1395":10,"14":[5,6,8,9,10,11],"1414":10,"1436":10,"1445":10,"145":[6,10],"1451":10,"1474":10,"1481e":10,"1491":10,"15":[6,8,9,10],"1511":10,"1532":10,"1541":10,"1548":10,"1570":10,"1587":10,"16":[6,10,11],"1607":10,"1628":10,"1637":10,"1643":10,"1660ti":7,"1666":10,"167":11,"1683":10,"17":[6,10,11],"1703":10,"1723":10,"1733":10,"1761":10,"1778":10,"1791e":10,"1797e":10,"18":[6,10,11],"180":[6,7,9],"1800e":10,"1818":10,"1828":10,"1872":10,"19":10,"1922":10,"19j":10,"1d":0,"1e":[0,6,7],"1e3":6,"1st":0,"2":[0,1,2,5,6,8,10,11],"20":[5,10],"2019":0,"2020":3,"21":[5,10],"219":[6,10],"22":10,"2257":9,"23":10,"230":11,"230000000000":0,"24":[0,5,10],"25":5,"2500":[9,10],"2582":9,"2599":9,"260":5,"288":9,"2898e":10,"2d":[0,8,10],"2x":5,"2x3":5,"3":[0,4,5,6,7,8,9,10,11],"300":10,"320":7,"3297":9,"33":[10,11],"334":11,"340":11,"35":8,"36":6,"3600":6,"3647603":3,"38":11,"384":11,"39":[6,9,10],"3blue1brown":5,"3d":[0,11],"3mm":11,"4":[0,3,5,6,7,8,9,10,11],"4011e":10,"42":6,"4232":9,"4302e":10,"431677053863599":6,"4392":9,"4498439":[7,9],"4581e":10,"47":[5,9],"480x480":[6,9],"4885e":10,"4891e":10,"4895e":10,"4899e":10,"4901e":10,"4902e":10,"4903e":10,"4930016":[6,8,10],"5":[5,6,7,8,9,10,11],"50":[9,10,11],"500":[6,9,11],"512":0,"5264e":10,"5266e":10,"5281":3,"54":5,"5509":9,"567":11,"569":9,"57":11,"571":9,"58":10,"5811":9,"5965":9,"5975":9,"6":[4,5,6,8,9,10,11],"600":[6,7],"6035":9,"6390e":10,"650160":6,"6559":9,"660567432744819":6,"7":[5,6,8,9,10,11],"7289":9,"7519":9,"7632e":10,"7633e":10,"767":11,"77":11,"7764":9,"7989":9,"8":[0,5,6,8,9,10,11],"800":[6,8,10],"8000":10,"8008":9,"804441297183736":6,"8154e":10,"8196e":10,"8408":9,"8421":9,"8480":9,"8584":9,"8628":9,"8657":9,"8916":9,"9":[5,6,7,8,9,10],"90":8,"9102":9,"9198":9,"9217e":10,"9292e":10,"9345":9,"9557":9,"9616":9,"9632":9,"9774":9,"9938":9,"9975":8,"9e7":7,"boolean":0,"break":[2,3,6],"case":[0,2,5,10,11],"catch":1,"class":[0,2,11],"default":[0,7,9,10,11],"do":[0,2,4,5,6,8,9,10,11],"export":[0,6,8,9,10],"final":[0,5,6,9,10],"float":[0,5],"function":[0,1,2,3,6,7,11],"import":[2,3,4,5,6,7,9,10],"int":[0,6,7,11],"long":[0,2,5,8,11],"new":[0,2,5,6,8],"null":0,"return":[0,5,6,7,9],"short":[2,7],"true":[0,5,6,7,8,9,10],"try":[6,7,8,10],"while":[2,5,6,7,8],A:[0,2,3,5,7,9,10,11],And:[0,2,6,8,10],As:[2,6,11],Be:2,But:[5,6,10],By:0,For:[0,2,3,5,6,7,8,9,10,11],If:[0,2,3,4,5,6,7,8,9,10,11],In:[0,2,5,6,7,8,9,10,11],It:[2,3,4,6,8,9,10,11],Its:0,No:2,Of:10,One:[2,5,6,7,8],The:[0,1,2,3,5,7,9],Then:[2,6,8,11],There:[0,2,3,5,6,8,11],These:[0,2,5,10],To:[0,2,3,5,6,7,8,9,10,11],With:[5,9],_:[0,11],__init__:[6,10],__version__:4,_build:2,_execution_engin:[6,10],_static:10,a8:11,a_tensor:5,ab:5,abil:[1,5,6],abl:[2,5],about:[2,3,5,6,8,10],abov:[0,2,5,11],absolut:[5,8],acceler:[3,6,10],accentu:9,accept:0,access:[0,4,8,10],accomplish:2,accord:0,account:[2,11],accur:[0,6,9],achiev:2,across:[0,11],act:0,action:2,activ:2,actual:6,ad:[1,2,5],adam:[6,7],add:[2,5,10],add_mask:0,add_scalar:6,addit:[0,2,3,6,7,10,11],addition:2,adequ:6,adjust:8,advanc:10,advers:6,advic:2,affect:6,after:[0,2,5,7,9,11],against:[0,6,9,10],aggreg:6,aim:2,aip:8,alert:8,algorithm:[6,9,10],all:[0,1,2,5,6,8,9,10,11],allevi:6,allow:[0,5],alma:[2,3,6,8,9,10,11],along:[0,11],alpha:[6,8,9],alreadi:[0,2,5,6,7,9,11],also:[0,2,4,6,7,8,9,10,11],alter:5,altern:2,alwai:[0,2,6],amax:9,ambiti:10,amin:9,among:0,amount:[0,10],amplitud:0,an:[0,2,3,5,6,7,8,9,11],an_arrai:5,analyt:[0,5],angl:[0,6,11],angu:5,ani:[0,2,3,6,7,10,11],anim:6,annoi:11,anoth:[5,7],another_arrai:5,another_tensor:5,answer:5,antenna:[6,11],anyth:2,apart:11,apertur:3,api:[3,8,10],appear:[0,5,8,10,11],append:[5,6,9,10],appendix:0,appli:[0,10,11],applic:[0,5,6,10],appreci:2,approach:6,appropri:[0,5,6,11],approv:2,approx:11,approxim:[0,6,11],approxm:0,ar:[0,2,4,5,6,7,8,9,10,11],arang:6,arbitrari:5,arcsec:[0,1,6,8,11],arcsecond:[0,11],arctan2:[6,11],ard:7,area:[0,7,11],areial:11,aren:[7,10],arg:0,argument:[0,2,11],aris:11,around:[0,6],arrai:[0,3,5,6,8,10,11],arrang:0,arriv:0,art:3,ascend:[0,5],aspir:0,assert:1,assertionerror:0,assess:[2,6],associ:5,assum:[0,2,6,7,8,11],assumpt:6,astronom:[8,11],astronomi:6,astrophys:[6,9],astropi:[2,6,7,8,9,10],atacama:3,atan2:11,aten:[6,10],attach:[0,2,8,10],attribut:[0,5],augment:0,author:3,auto:3,autodifferenti:[5,11],autodifferentiaton:5,autograd:[5,6,10],autom:[2,5],automat:[0,2,5,10],avail:[0,2,4,7,10],averag:[0,6,8],avoid:[2,8],ax:[6,8,9,10,11],axessubplot:6,axi:[0,11],azimuth:[0,6],b:[0,2,7],back:[2,10,11],background:0,backward:[5,6,7,9,10,11],bake:6,band:11,base:[0,5,9,10,11],base_cub:[0,9,10],basecub:[0,6,9,10],baselin:[0,8,10,11],basetemp:2,basi:10,basic:[0,7,10],bayesian:6,bcube:[0,9,10],bcube_numpi:10,bcube_pytorch:10,beam:[0,6,8,9,11],beam_kwarg:0,becaus:[0,6,8,10,11],becom:2,been:[0,2,6,7,8],befor:[5,6,8,9,10,11],begin:[2,9],behind:2,being:[0,2,6],believ:10,below:[2,3],benefit:5,best:[5,6,10],better:[6,9,10],between:[0,6,8,9,10,11],beyond:11,bia:[9,10],big:[6,9],bin:[0,2],bit:10,bl:0,blank:[9,10],blog:5,blow:5,bool:0,borrow:7,both:[0,7,11],bottom:[0,5,6,8],bound:10,boundari:0,bracewel:[0,11],branch:2,breviti:0,brianna:[1,3],brigg:[0,1,6,8,9,11],bright:[0,11],broadcast:0,browser:2,bug:[2,10],build:[1,3,6],build_grid_mask_from_cel:0,built:[2,3,5],bundl:[1,10],c1:5,c43:6,c:[5,6,7],cach:[6,7,8,9,10],cal:11,calcul:[0,3,6,7,8,9,10],calculu:5,calibr:8,call:[0,6,8,10,11],can:[0,2,4,5,6,7,8,9,10,11],cannot:[0,2,5],capabl:3,capac:8,card:[4,7],care:9,carri:[0,2,10,11],cartesian:11,casa:[0,6,8,9],cast:[6,10],cd:[2,4],cdelt1:6,cdelt2:6,cell:[0,2,6,8,9,11],cell_index_list:0,cell_siz:[0,1,6,7,8,9,10,11],center:[0,5,6,11],central:[2,6,7,9,10],central_chan:7,centroid:0,certain:[6,8,10,11],chain:5,chan:[6,8,9,10],chang:[1,2,5,7,8,9,10],changelog:3,channel:[0,6,7,8,9,10,11],chapter:[8,11],characterist:10,chart:[1,2],check:[0,2,3,5,6,7,10,11],check_data_fit:0,check_visibility_scatt:[0,8],checkout:2,chi:[0,6,10],choic:[0,5,6],choos:[1,3,5,11],chose:[5,10],chosen:5,chunk:[0,6],citat:1,cite:3,clean:[2,6,8,10],clear:11,clearli:9,cli:2,click:2,clone:4,close:[5,6,9],closer:9,cluster:6,cmap:[6,9],co:[0,6,8,9],coars:1,code:[0,3,7,10,11],codebas:[2,11],collabor:2,collaps:0,color:[5,6],colorbar:[9,10],colormap:9,column:11,com:[2,4],combin:6,come:[0,6],command:[2,6,7],commit:2,common:[5,7,10,11],commonli:[0,11],commun:[6,10],compar:[0,2,5,6,8,10],comparison:[6,10],complet:[2,5,6,7,9,10],complex128:10,complex:[0,5,6,8,9,10],compli:2,complic:[8,11],compon:[0,10,11],compos:0,comprehens:2,compromis:8,comput:[0,5,6,7,10],concept:[5,6,10],concern:6,conclus:3,confid:6,config:[2,6,7],configur:[2,3,6,9],confirm:8,conflict:2,confus:11,congratul:7,conj:8,conjug:[0,6,8,9],connect:[0,10],connector:[3,6],consid:[0,5,6,11],consider:11,consist:5,constant:[0,10],constel:3,constraint:10,construct:[0,10],consult:[6,9,11],contain:[0,2,5,6,7,8,9,10],content:2,context:[6,11],continu:[3,5,6,10],continuum:[0,3,8],contrast:9,contribut:[3,4,10],control:2,conv_lay:[9,10],convei:8,conveni:[0,8,11],convent:[0,2,3,8,9],converg:10,convers:[0,10],convert:[0,2,6,8,9,10,11],convolut:0,coord:[0,6,7,8,9,10],coordin:[3,5,6,7,8,9,10,11],copi:[0,2,6,7,9,10],core:[0,10],correct:[2,8,11],correctli:[2,7,8,10,11],correl:0,correspond:[0,2,6,8,9,10,11],cosin:11,could:[0,2,5,10],count:[0,10,11],cours:[2,6,9,10],cov:2,cover:[2,8,9,10],coverag:6,cpp:[6,10],cpu:[7,9,10],creat:[0,2,5,6,7,8,9,10,11],criterion:5,cross:[1,3,7,9],cross_valid:6,cube:[0,1,6,8,9,10],cuda:3,current:[0,2,4,5,9,10,11],cv:[0,6],cycl:11,czekala:3,d:[3,4,5,6,7,8,9,10,11],d_:0,danger:5,daniel:[8,11],dark:5,dartboard:[0,1,6],data:[0,2,3,5,6,7,9,10,11],data_im:[0,6,7,8,9,10],data_r:[0,6,7,8,9,10],data_vi:0,dataconnector:0,datapoint:0,dataset:[1,2,3,5,6,7,8,9],datasetconnector:1,datasetgrid:0,date:2,deal:11,debug:[0,8],dec:[0,6,11],decim:6,deconvolut:9,decreas:11,deep:10,def:[5,6],defacto:1,defalt:0,defin:[0,5,6,8,9,10,11],definit:0,deg:6,degre:0,delet:2,deliv:[0,8],delta:[6,8,9,11],delta_i:0,delta_l:0,delta_m:0,delta_x:0,demonstr:[6,7,9,10,11],densiti:[0,1,6],depend:6,deriv:[0,5,8],descent:3,describ:[2,6,8,9,10,11],descript:0,design:[0,6,7,10],desir:[0,10],detach:[6,9,10],detail:[0,7],determin:[4,5],dev:[2,3,4],develop:[3,4],deviat:0,devic:[0,7],dft:11,diagnost:[3,9,10,11],diagram:10,dict:0,dictionari:[6,7,9],did:6,diff:2,differ:[2,6,8,9,10,11],differenti:[3,5],difficult:8,dimens:[0,6,8,9,10],dimension:[0,8,10,11],direct:[0,2,5,10,11],directli:[0,8],directori:2,dirti:[0,3,6,8,11],dirty_imag:[1,9],dirty_image_model:9,disabl:8,discard:[6,10],discret:3,discuss:[2,3,10],disk:8,displai:[9,10,11],dissimilar:10,distanc:9,distribut:[6,8,11],diverg:[5,9,10],dl:0,dm:0,doc:[1,2],docstr:2,document:[0,3,4,5,8,10],doe:[6,9,11],doesn:[2,6,10],doi:3,domain:[0,11],domin:10,don:[0,2,6,8,11],done:[0,2,6,9],doubl:0,down:[3,5,7,8,11],download:[2,4,7],download_fil:[6,7,8,9,10],downward:5,draft:2,draw:[2,6],drawn:[6,11],dset:[6,7,8,9,10],dtype:[9,10],du:0,dual:0,due:[2,6],durat:6,dure:0,dv:0,dx:5,dy:5,e:[0,2,8,9,11],each:[0,2,5,6,8,11],easi:[0,5,10],easier:2,easili:[5,10,11],east:[0,11],edg:0,edge_clamp:0,edit:2,effect:[0,11],effici:5,eht:[0,1],either:[0,7,8],elapsed_tim:7,element:[0,11],ell:0,els:7,emiss:[6,9],emphasi:2,empir:10,empti:7,empty_cach:7,enable_tim:7,enclos:0,encourag:0,end:[0,6,7,10],enforc:[0,1,10],enorm:5,enough:[6,9,10],ensur:[0,2,7],entir:9,entropi:[0,1],enumer:6,environ:[3,8],envis:2,epoch:[6,7],epsilon:0,eqn:11,equal:[0,5,6],equat:[0,5,11],equival:[0,6,11],error:[2,9,10],especi:[2,6,7],essenti:0,estim:[0,10],etc:10,euclidian:9,evalu:[0,5,6,10],even:[0,2,6,8,9,10,11],event:7,everi:2,everyth:7,evid:6,exact:7,examin:10,exampl:[0,2,5,6,7,10,11],exce:10,excel:2,excit:10,execut:2,exercis:5,exist:[0,2],exoplanet:2,exp:[0,11],expect:[0,5,6,9,10,11],experi:10,explain:2,explicitli:11,explor:6,express:[8,11],ext:6,extend:3,extens:2,extent:[0,6,8,9,10],extra:[0,2,6],extra_requir:2,extrem:10,f:[0,5,11],f_0:0,f_1:0,f_2:0,f_3:0,f_:[0,6],face:11,fact:11,factor:0,fail:2,fals:[0,6,8],familiar:[2,6,8],far:5,fast:11,favor:1,favorit:2,fcube:0,fear:2,featur:2,feb:3,fed:0,feed:[0,8,10],feedback:2,feel:[5,10],fell:0,felt:5,fetch:2,few:[6,10],fewer:9,fft2:11,fft:0,fftshift:[0,11],fftspace:0,fidel:6,field:11,fig:[6,8,9,10],figsiz:[6,8],figur:[2,6,9,11],file:[0,2,6,7,8,9,10],filenam:10,find:[2,5,6,8],finish:2,first:[0,2,5,6,7,8,9,10,11],fit:[0,6,9,10,11],fix:[0,2],flat:5,flatten:6,flayer:6,flexibl:3,flip:[0,11],float64:[0,9,10],flowchart:1,flux:9,fname:[0,6,7,8,9,10],focu:3,fold:[0,3],folder:2,follow:[0,1,2,5,6,7,8,10,11],form:[0,2,5,6,10],format:[0,2,6,10],forward:[0,6,7,8,9,10,11],found:[2,4,9],fourier:[0,3,6,8],fourier_gaussian_klambda_arcsec:0,fourier_gaussian_lambda_radian:0,fouriercub:[0,6],frac:[0,5,11],framework:[3,5],free:[2,10],frequenc:[0,1,6,11],frequent:8,from:[0,3,5,6,7,8,9,10,11],full:[0,2,4,6,10],fulli:[0,6,11],functionil:0,fundament:10,further:10,g:[0,2,3,8,11],gap:6,gaussian:0,gener:[0,2,6,7],geometr:11,get:[3,5,6,7,10],get_dirty_beam:0,get_dirty_beam_area:0,get_dirty_imag:[0,1,6,8,9,11],get_jy_arcsec2:0,get_max_spatial_freq:0,get_maximum_cell_s:0,get_nonzero_cell_indic:0,get_polar_histogram:0,ghz:11,git:[2,4],github:[1,3,4,10],give:2,given:[0,5],gnbu:6,gnu:2,go:[0,2,5,7,11],goal:[0,2,6,10],goe:8,good:[5,8],gpu:[3,4,6,10],grad:[5,7,10],grad_fn:10,gradient:[0,3,6],graph:[5,10],graphic:[4,7],great:[9,10],greater:0,greatli:[0,9],grid:[3,6,7,9,11],gridcoord:[0,1,3,6,7,9,10],griddeddataset:[0,6],griddedresidualconnector:[0,6],gridder:[0,1,2,3,6,7,9,10,11],ground:[0,5,11],ground_amp:0,ground_cub:0,ground_cube_to_packed_cub:[0,1,11],ground_mask:[0,6],ground_phas:0,ground_residu:0,ground_vi:0,gt:[6,9,10],gtx:7,guess:9,gui:7,guid:2,guidanc:2,guidelin:2,ha:[0,2,7,8,9,10,11],had:[6,11],half:6,hand:5,handi:6,hann:[0,1],hannconvcub:[0,1],happen:10,hard:6,hardwar:7,hast:6,have:[0,2,3,4,5,6,7,9,10,11],haven:5,hdul:6,header:[0,2,6],header_kwarg:0,help:[2,3,6,10],henc:[2,8],here:[2,5,6,7,8,9,10,11],hermitian:[0,6],high:6,highlight:[5,9],hill:5,histogram:0,histor:8,hogg:6,hold:0,hole:6,home:2,homepag:4,honest:10,hook:2,hopefulli:10,host:[2,3,4,6],hostedtoolcach:[6,10],hour:6,how:[0,2,3,5,6,7,8,9,10,11],howev:[2,5,6,8,9,11],hspace:6,html:2,htmlcov:2,http:[2,3,4,6,7,8,9,10],hundr:10,hyperparamet:6,hz:0,i:[0,2,6,7,8,9,10,11],i_:[0,11],i_i:0,ian:3,iancz:3,icub:[0,6,7,9,10],idea:[0,2,6,8,10],ideal:6,ident:[0,11],identifi:6,ifft:11,ifftshift:11,ignor:6,ill:[10,11],im:[0,6,9,10],imag:[1,3,5,6,7],imagecub:[0,6,9,10],imaginari:[0,6,10],imax:9,img:[6,8,9],img_ext:[0,6,8,9,10],imin:9,implement:[0,1,2,6,8,9,11],implicitli:[0,6],implment:11,impos:9,improv:[2,6,10],imshow:[0,6,8,9,10,11],incas:7,inclin:5,includ:[0,1,2,5,6,11],incorpor:[1,2,6],incorrect:8,incorrectli:8,increas:[0,9,11],ind:6,independ:5,index:[0,2,3,11],index_vi:0,indic:[0,11],individu:[0,2,6,10],infer:[6,8],info:[7,9],inform:[0,2,3,4,7,8,9,10,11],infti:11,inher:6,inherit:0,initi:[0,2,3,5,6,7,10],input:[0,7,8,11],instal:[1,3,5,8],instanc:0,instanti:[0,8],instead:[1,5,6],instruct:[2,7],int_:11,integ:0,integr:[2,11],intens:[0,9,11],intent:2,interest:[2,4,6,8],interfac:2,interferomet:[3,11],interferometr:[0,6,11],interferometri:[3,8],intermedi:[6,11],intern:[0,6,10,11],interpol:[6,8,9,10],interpret:[7,8],introduc:[1,2],introduct:3,introductori:10,invalid:5,invers:[10,11],invert_xaxi:6,investig:2,invoc:2,invok:2,involv:0,io:[1,6],ipynb:2,ipython:10,is_avail:7,isn:[2,7,10],issu:[0,2,4,8,10],item:[5,6,9,10],iter:[0,3,5,6,7,8,9],its:[0,2,4,5,8,9,11],itself:[0,2,6,10],iv:[0,1],j:[0,6,11],janski:3,jax:5,jean:0,job:6,join:3,js:[1,2],jupyt:[1,2],jupytext:[1,2],just:[0,2,6,7,8,10,11],jy:[0,1,8,9,11],k:[0,3,11],k_fold:6,k_fold_dataset:6,karl:3,keep:[0,2,5,6,10,11],kei:[2,5],kept:5,kernel:[0,2,3],keystrok:0,keyword:[0,8],kfoldcrossvalidatorgrid:[0,1,6],kilolambda:0,kit:4,klambda:0,km:11,know:[2,5,10,11],known:9,kw:[6,8,9],kwarg:0,l2:9,l:[0,11],l_1:0,l_2:0,label:6,laid:[0,8],lambda:[0,6,11],lambda_:6,lambda_spars:[6,7],lambda_tv:[6,7],land:0,laplac:5,larg:[3,5,6,7,10],large_step_s:5,larger:[0,5,9],last:[0,5],lastli:10,later:[5,10],latest:4,layer:0,layout:0,lead:0,learn:[2,5,6,10],least:[0,6],leav:11,left:[0,5,6,7,8,11],leftrightharpoon:0,len:6,length:[0,11],lens:5,less:8,let:[0,5,6,9,10,11],level:6,lib:[6,10],lieu:0,lifecycl:2,like:[0,1,2,3,4,5,6,7,8,9,10,11],likelihood:[0,3,5,9,10],limit:[0,2,6],line:[2,3,5,8],linear:0,linearli:0,linewidth:6,link:[2,7,11],linspac:5,linux:[7,8],list:[0,2,5,6],literatur:10,ll:[0,5,6,7,8,9,10],lmv:0,ln:0,load:[3,6,7,8,10],load_state_dict:9,local:2,locat:[0,2,5,6],log:[0,2],log_stretch:0,logarithm:[0,1],loglinspac:0,logo:[6,8,9,10],logo_cub:[6,7,8,9,10],logspac:0,longer:[0,5,10],look:[0,2,3,5,6,8,9,10,11],loomi:[1,3],loop:[3,4,5,7],loos:[0,8],loss:[1,3,6,7],loss_track:[9,10],lossfunc:9,lot:2,love:2,low:6,lower:[0,6,8,9,10,11],lr:[6,7,9,10],lt:[6,9,10],ly:[6,10],m:[0,2,9,10,11],m_:0,m_linear:0,machin:[2,6,10],maco:8,made:[1,2,8],magic:5,mai:[0,2,4,7,8],main:[0,2,7],mainli:[0,8],make:[2,5,6,7,8,9,11],makefil:2,mani:[0,2,5,6,9,10,11],manner:[0,10],map:[0,6],mask:[0,3,6],match:0,materi:2,mathemat:[5,11],mathrm:[0,1,6,11],mathtt:11,matmul:5,matplotilb:11,matplotlib:[0,5,6,8,9,10,11],matric:5,matrix:5,max:[0,6],max_freq:0,max_grid:0,max_scatt:[0,8],maximum:[0,3,5,8,9,10],mcmc:6,mean:[0,2,5,6,9,10,11],meant:8,measur:[0,6,8,11],mechanist:5,memori:7,merg:2,mermaid:[1,2],meta:10,metadata:2,meter:11,method:[0,5,6,7,8,10,11],metric:6,metropoli:6,middl:0,might:[0,2,5,6,10,11],migrat:1,mileag:7,millimet:3,millisecond:7,minim:[0,10],minimum:[0,5,10],mirror:8,miss:[5,6],mix:0,mjy:6,mkdir:2,mmd:10,mmdc:2,mock:[2,6,8,9,10],mode:[6,7],model:[0,3,6,7,8,11],model_vi:0,modifi:2,modul:[1,3,7,10],modular:8,moment:5,monolith:8,month:3,moran:8,more:[0,2,3,4,5,6,7,8,9,10,11],most:[0,5,6,7,8,10,11],mostli:6,move:[0,1,7,8],mpol:[0,1,4,5,6,8,9,10,11],mpoldataset:[2,6,8],ms:[8,9],mseloss:9,much:10,mulbackward0:10,multi:[0,2,6,7,9,10],multipl:[0,5,9,11],multipli:5,multivari:5,must:[0,7,9,11],my_feature_branch:2,mycoord:0,n:[0,7],n_log:0,n_v:0,nabla:5,name:[0,2],narrow:10,nativ:[0,1,6,10],natur:[0,1,8],naxis1:6,naxis2:6,nbsphinx:2,nchan:[0,6,7,8,9,10,11],ncol:[6,8,9],nearli:[0,5,6,8,11],neat:2,necessari:[2,10],need:[0,2,4,5,6,7,8,9,10,11],neg:[6,9],network:[0,4,7,10],neural:[4,7,10],never:0,newer:2,next:[0,6,7],nll:[0,6],nll_grid:[0,1,6,7,10],nn:[0,9],node:0,nois:[6,8,10],non:[0,6,11],none:[0,6,8,9,10],norm:[0,6,9,11],normal:[0,2,6,8,9,10,11],north:[0,11],note:[0,2,4,5,8,9,10],notebook:[1,2],noth:5,notic:2,now:[1,5,7,9,10],np:[0,5,6,7,8,9,10],npix:[0,6,7,8,9,10,11],npseed:[0,6],npy:8,npz:[6,7,8,9,10],nrow:[6,9,10],ntheta:0,nu:0,number:[0,6,8,11],numer:[8,11],numpi:[0,5,6,7,8,9,10,11],nvi:[0,6,8],nvidia:[4,6,7,10],nx:6,ny:6,nyquist:0,object:[0,1,3,6,7],observ:[3,6,8,11],obtain:[5,8],occasion:11,off:9,offend:2,offici:[2,7],offset:0,often:[6,10],ok:[8,11],old:7,omega:0,onc:[0,2,6,7,9],one:[0,2,5,6,7,8,9,10],ones:[7,9,10],onli:[0,2,5,6,7,8,9,10,11],open:[0,2,6,7],oper:[0,5,7,11],opinion:8,opportun:11,oppos:[0,9],opt:[0,6,10],optim:[0,1,3,4,6,7],option:0,orang:5,order:[9,10,11],ordereddict:[9,10],orderli:2,org:[3,4,6,7,8,9,10],organ:[2,3,10],orient:[10,11],origin:[0,2,6,8,9,10,11],other:[2,3,5,6,7,8],otherwis:[0,2,5,10],our:[2,3,5,6,7,8,9,10],ourselv:[8,10],out:[0,2,3,6,8,10,11],outdat:2,outlin:10,output:[0,1,2,3,5,11],over:[0,5,7,11],overlap:0,overwrit:0,own:[0,2,6,8],p:0,p_i:0,pack:0,packag:[0,1,2,3,4,5,6,8,10,11],packed_cub:0,packed_cube_to_ground_cub:[0,1,11],packed_cube_to_sky_cub:[0,1,10,11],page:[0,1,2,3,4,7,11],pair:[0,2,6],pandoc:2,paperspac:5,par:6,paramet:[0,6,7,8,9,10,11],parameter:[0,9,11],paramt:9,parlanc:10,part:[0,2,3,5,6,8,9,10],particular:[4,7],partit:[0,6],pass:[0,2,5,7,8],passthrough:0,pasthrough:0,pattern:[2,6],peak:0,penalti:0,per:[0,11],perform:[0,5,6,7],perhap:2,permit:[0,9],permut:0,perspect:11,ph:[8,11],phase:[0,3],phi:[0,6,11],phi_cel:0,phi_edg:0,phrase:0,pi:[0,11],pick:5,piecewis:0,pip:[1,2,3,7],pitfal:11,pixel:[0,1,6,9,10],pixel_map:0,pkgname:[6,8,9,10],place:[2,5,6],plane:[0,6,8,9,11],pleas:[0,2,3,4,6,9,10],plot:[0,5,6,9,10,11],plotsdir:2,plt:[5,6,8,9,10],plug:5,plugin:2,png:2,point:[0,5,6,9,10,11],polar:[0,11],popul:5,popular:10,posit:[0,1,5,9,10,11],possibl:[0,2,6,8,9,10],possipl:9,post:[2,5],posterior:6,potenti:[3,4,6],power:[0,1,6,7],practic:11,pre:[0,2],preced:2,precis:8,precompos:[1,3,6,7,9,10],predict:[6,10],prefactor:[0,6,11],preliminari:1,premultipli:0,prepack:0,prepar:2,preprocess:2,present:[6,10],preserv:[0,2],pretti:5,preview:2,previou:[5,6,7,10],previous:0,prime:[0,6,8,9],primit:10,principl:2,print:[4,5,6,7,8,10],prior:[0,5,9],prior_intens:0,probabl:10,probe:[2,6],problem:[3,9,10],procedur:11,proceed:5,process:[0,1,2,5,6,7,9,10,11],prod_arrai:5,prod_tensor:5,produc:[0,2,6,8,11],product:[0,2,5,10,11],program:[2,5],progress:2,project:2,prolat:1,promot:0,propag:[8,10],proper:5,properli:[2,11],properti:[0,8],propos:11,provid:[0,2,3,5,6,7,8,9,10,11],psd:0,psf:[0,8],pt:9,publicli:2,publish:3,pull:[0,2,4,10],purpos:[0,6,7,8,9,10,11],push:2,put:6,py:[2,6,10],pypi:[2,4],pyplot:[0,5,6,8,9,10,11],pytest:2,python3:[2,6,10],python:[3,4,6,8,10],pytorch:[0,3,4,6,8,9,11],q:[0,6,11],q_:0,q_cell:0,q_edg:0,q_max:0,qs:0,quantiti:[0,11],queri:2,question:[3,10],quick:[0,2,7,11],quickli:[5,6],quit:[6,8,11],r:[0,5,6,8,9,10,11],ra:[0,6],radial:[0,6],radian:[0,11],radio:8,rai:[6,11],rais:[0,2,4,8,10],random:[0,6,9],randomli:6,rang:[0,2,6,7,9,10,11],raster:6,rate:[5,10,11],rather:9,rayleigh:0,re:[0,2,4,5,6,9,11],reach:[2,5,6,9,10],read:[1,2,8],readi:[2,7,9],real:[0,6,8,10,11],realist:10,realli:[6,8],reason:[0,5,10,11],rebuild:2,recap:3,recent:8,recogniz:11,recommend:[0,2,6,8],recommit:2,reconstruct:3,record:[6,7,8,9,10],recov:6,redo:2,reduct:9,refer:[0,2,9,11],reflect:[2,10],region:0,regular:[0,3,5,6,8,9,10],rel:[0,1,5,8],relat:11,relationship:[0,11],releas:3,relev:10,reli:2,reliabl:2,reload:9,remain:[6,9],rememb:[2,8],reminisc:0,remot:2,remov:[1,10],renam:2,render:[2,5],repeat:[2,5],replac:6,replic:6,replot:5,repo:2,report:2,repositori:[2,3,4],repres:[0,9,10,11],represent:[0,3,9],reproduc:0,request:[0,2,10],requir:[0,2,4,7,8,10,11],requires_grad:[0,5],requri:7,research:3,resembl:9,reset:9,residu:0,resolut:8,resourc:[2,3],respect:[0,5,10,11],rest:[7,10],restart:7,restrict:[0,8],result:[0,3,5,7,9,10],retain:5,review:2,revisit:6,rfft:0,rhel:8,right:[0,2,5,8,11],rigor:2,rml:[0,3,5,6,8,9,10,11],robust:[0,1,6,8,9],role:10,root:[0,2],rotat:[0,6],rough:0,routin:[0,1,8,10,11],row:11,rst:2,rtest:6,rtrain:6,rule:5,run:[0,7,9],run_backward:[6,10],runtimeerror:[0,8,10],runtimewarn:[0,8],ryan:[1,3],s:[0,2,4,5,6,8,9,10,11],sai:[5,11],same:[0,5,6,8,9,11],sampl:[0,6,11],sampler:6,satisfi:2,save:[0,2,8,9],saw:10,scale:[0,5,8],scatter:[0,5,6,8,10],scene:2,scheme:[6,8],scope:2,score:[6,7],scratch:6,script:[0,2],scroll:7,second:[5,7],section:7,see:[0,2,4,5,6,7,8,9,10,11],seed:0,seen:[4,9,11],self:[0,2],semin:11,sens:[3,6,8,9],sensit:8,separ:[4,6,8],sequenc:11,serv:11,session:2,set:[0,3,5,6,7,8,9,10,11],set_cmap:9,set_ticklabel:6,set_titl:[6,8,9],set_xlabel:[6,8,9,10],set_xlim:6,set_ylabel:[6,8,9,10],set_ylim:6,setup:[2,3,10],sever:[2,5,6,8,10,11],sgd:[9,10],shape:[0,6,8,10,11],shell:2,shift:[0,11],ship:2,shot:8,should:[0,2,6,7,8,9,10],show:[5,6,7,9,10],show_progress:[6,8,9,10],shown:[5,9],side:[0,5],sigma:0,sigma_i:0,sigma_l:0,sigma_m:0,sigma_x:0,signal:10,signific:[0,2],significantli:6,similar:[0,5,6,8,10],simpl:[0,7,8,10,11],simplenet:[0,1,6,7,9,10],simplest:10,simpli:0,simplic:0,simplifi:0,simul:6,simultan:0,sin:0,sinc:[0,2,5,9,10,11],singl:[0,5,6,8,9,10,11],single_channel_estim:0,site:[6,7,10],situat:11,size:[0,5,6,8,9],sizeabl:6,skip:7,sky:[0,6,10,11],sky_cub:[0,6,7,9,10],sky_cube_to_packed_cub:[0,1,11],sky_gaussian_arcsec:0,sky_gaussian_radian:0,sky_model:0,skycub:7,slice:[0,6],slightli:6,slope:5,slope_curr:5,slope_large_step_curr:5,slope_start:5,small:[2,5,6,10,11],smaller:9,smooth:0,snp:9,so:[2,5,6,8,9,10,11],soften:0,softplu:0,softwar:[2,3,7],solut:[2,9],solv:5,some:[0,2,5,6,10,11],someon:2,someth:[0,11],sometim:6,somewhat:6,somewher:5,sophist:6,sourc:[0,2,3,7,9,10,11],south:0,space:[0,6,8,10,11],span:0,spars:[0,3,6],sparsiti:[0,6,7],spatial:[0,1,6,11],special:2,specif:[0,2,4,7],specifi:[0,2,8,11],spectral:[0,1,3,8,9],sped:[6,10],speed:[0,5,7],spheroid:1,spheroidal_grid:1,sphinx:2,split:[0,6],spot:10,spread:0,sqrt:[0,6,9,11],squar:[0,5,9],squareroot:9,squeez:[6,9,10],src:[6,10],stabl:[2,4],stale:2,stand:5,standard:[0,6,8,11],start:[0,2,3,5,6,7,9,10],state:[0,3,9,10],state_dict:[9,10],statement:[1,7],statist:8,steepest:5,step:[0,2,5,6,7,9,10,11],step_siz:5,steradian:0,still:[0,2,5,6,8,10],stochast:10,stop:[2,5],store:[0,1,2,5,6],str:0,straightforward:8,strang:5,strength:[0,6],stretch:0,strictli:2,string:0,strive:2,strong:[6,7,9],structur:6,submillimet:3,submit:[2,10],subpartit:0,subplot:[6,8,9,10],subplots_adjust:[6,8],subscript:11,subsect:2,subselect:6,subtleti:11,success:2,successfulli:[2,9],suffici:10,suggest:[2,6],suit:[2,10],sum:[0,5,6,9],sum_:[0,11],sum_i:0,summarywrit:6,superset:2,suppli:0,support:[0,3,8],sure:[2,5,6,9],svg:10,swap:10,sweep:6,swenson:8,sy:[9,10],symmetr:0,sync:2,synchron:7,synthes:[10,11],system:[2,4,7,11],t:[0,2,5,6,7,8,9,10,11],t_b:0,tabl:[2,8],tag:2,take:[0,2,3,5,8,9,10,11],taken:9,tangent:5,taper:0,taper_funct:0,tclean:[0,6,8,9],technic:[0,7],techniqu:6,tediou:2,tell:6,temperatur:0,temporari:2,tensor:[0,3,6,7,9,10],tensorboard:6,tensorvis:4,term:[0,5,6,10,11],test:[0,3,6],test_chan:6,test_dset:6,test_mask:6,test_scor:6,text:[5,6,7,9,10],textbook:8,th:[0,6],than:[0,2,5,6,7,11],thank:[2,11],thankfulli:11,thei:[0,2,5,6,9,10,11],them:[0,2,6,8,9],themselv:[0,8],theorem:0,therefor:[0,11],thermal:[0,8],thesi:[8,11],thi:[0,1,2,4,5,6,7,8,9,10,11],thing:[2,5,8,10],think:[0,2],thompson:8,those:[0,2,6,8,10],though:[0,2,8,9,10],thought:[6,11],three:[0,8,11],through:[0,2,6,9,10],throughout:[2,11],thu:0,time:[2,5,6,7,9,10,11],titl:3,tm:[8,9,11],to_fit:0,to_pytorch_dataset:[0,6,7,8,9,10],togeth:[0,5,6,10],told:2,too:[1,2,5,6],took:7,tool:[2,4,6,8],top:[0,3,8],topic:2,torch:[0,5,6,7,9,10,11],torchvis:7,total:[0,6,9,10],toward:[5,9,10],track:[5,10],train:[0,3,6,7],train_and_imag:6,train_chan:6,train_dset:6,train_mask:6,trainabl:0,transform:[0,3,8],translat:[0,11],trial:10,tricki:8,trigger:[6,10],troubl:[2,4],tune:[0,6,10],tupl:0,turn:[0,6,9],tutori:[1,4,5,6,7,8,9,10],tv:[0,6],tv_channel:0,tv_imag:[0,6,7],tweak:6,twice:10,two:[0,2,7,8],type:[0,6,11],typic:[6,11],typo:2,u:[0,6,11],u_bin_max:0,u_bin_min:0,u_cent:0,u_edg:0,uk:6,ul:11,uncertainti:[8,10],undefin:0,under:2,understand:5,unfamiliar:[2,9],ungrid:0,uniform:[0,8,9,10],uniniti:9,uniqu:8,unit:[0,1,3,8,9],unless:[0,8],unlik:10,unnorm:6,unpack:[0,11],unprim:0,unregular:9,until:[2,5,7],untouch:11,up:[0,3,5,6,7,9,10,11],updat:[1,2,4,5,6,7,10],upgrad:3,upload:2,upon:2,upsid:8,upstream:2,url:3,us:[0,1,2,3,5,6,8,9,10,11],useabl:2,user:[1,2,8,9,10,11],userwarn:[6,10],usual:[6,8,9,11],util:[2,3,4,6,7,8,9,10,11],uu:[0,6,7,8,9,10],uu_vv_point:0,uv:[0,6],uv_spars:0,uvdataset:0,v0:[2,3],v:[0,6,11],v_:11,v_bin_max:0,v_bin_min:0,v_center:0,v_edg:0,va:5,val:6,valid:[1,3,7,8,9],vallei:5,valu:[0,1,2,5,6,9,10,11],valuabl:2,vanilla:2,vari:[6,7],variabl:[0,2,5,6,7,10],variat:0,ve:[2,5,6,7,9,10],vector:[0,7],veloc:0,venv:2,veri:[2,3,5,6,8],verifi:2,version:[2,3,4,8,11],vi:[0,6,7,10],via:[0,2,4,6,8,9,11],video:5,view:[0,11],villar:6,violat:0,viridi:9,vis_ext:[0,6],vis_grid:0,vis_index:0,visibilit:10,visibl:[0,6,8,9,10,11],visit:7,visual:[0,3,6,9,11],vk:6,vla:[3,11],vm:11,vmax:[6,9],vmin:[6,9],vv:[0,6,7,8,9,10],w:[0,10],w_i:0,wa:[0,7,9,11],wai:[2,5,6,8,11],walk:10,wall:[9,10],want:[0,2,5,6,8,10,11],warn:8,warrant:11,wavefunct:1,wavelength:11,we:[0,2,3,5,6,7,8,9,10,11],web:2,websit:3,wedg:0,weight:[0,1,3,6,7,9,10],weight_grid:0,weight_index:0,weight_indexd:0,well:[2,6],went:2,were:[5,6,8,10],west:0,what:[0,2,5,6,10],whatev:2,when:[2,5,6,8,9,11],where:[0,5,6,11],whether:[0,2,6,7,8],which:[0,1,2,3,5,6,7,8,9,10,11],whole:6,whose:2,why:[2,3,5,6,8],width:0,william:5,window:[0,2,7],witheld:6,within:[0,2,5,11],without:[2,5,6,10],won:[6,10],work:[0,2,3,5,6,7,8,9,10],workflow:[2,6,10],world:11,worri:[2,6],worth:4,worthi:2,worthwhil:10,would:[0,2,5,6,11],wouldn:[5,6],wrap:[6,10],wrapper:0,wrapup:3,write:[0,2,6],writer:6,written:[2,9],wrong:2,wspace:[6,8],x64:[6,10],x:[0,5,11],x_:5,x_coord:5,x_current:5,x_input:5,x_large_coord:5,x_large_step:5,x_large_step_curr:5,x_large_step_new:5,x_new:5,x_start:5,xaxi:6,xlabel:5,xlim:5,xmax:5,xmin:5,y:[0,5],y_coord:5,y_current:5,y_large_coord:5,y_large_step_curr:5,y_large_step_new:5,y_new:5,y_start:5,yaml:2,yaxi:6,ye:5,year:3,yet:10,yield:10,ylabel:5,ylim:5,ymax:5,ymin:5,you:[0,2,3,4,5,6,7,8,9,10,11],your:[0,3,4,5,6,7,8,10],yourusernam:2,zawadzki:[1,3],zenodo:[2,3,6,7,8,9,10],zero:[0,5,6,7,8,10],zero_grad:[6,7,9,10],zeroth:11,zorder:5},titles:["API","Changelog","Developer Documentation","Million Points of Light (MPoL)","Installation","Introduction to PyTorch Tensors and Gradient Descent","Cross validation","GPU Acceleration","Gridding and diagnostic images","Initializing Model with the Dirty Image","Optimization Loop","Units and Conventions"],titleterms:{"0":1,"1":1,"4":1,"5":1,"break":10,"function":[5,9,10],"import":8,The:[6,8,10,11],acceler:[4,7],addit:5,an:10,angular:11,api:0,best:2,build:[2,10],cach:2,calcul:5,changelog:1,check:8,choos:6,clone:2,code:2,conclus:9,configur:7,connector:0,continu:11,contribut:2,convent:11,coordin:0,coverag:2,cross:[0,6],cube:11,cuda:[4,7],data:8,dataset:[0,10],debug:2,depend:2,descent:[5,10],develop:2,diagnost:8,dimens:11,dirti:9,discret:11,document:2,down:10,environ:2,fft:11,flux:11,fold:6,fork:2,fourier:11,from:[2,4],github:2,gpu:7,gradient:[5,10],grid:[0,8,10],gridcoord:8,gridder:8,guid:3,imag:[0,8,9,10,11],initi:9,instal:[2,4,7],interferometri:11,introduct:5,iter:10,k:6,light:3,load:9,loop:[6,9,10],loss:[0,9,10],million:3,model:[2,9,10],modul:0,mpol:[2,3,7],object:8,optim:[5,9,10],output:10,pack:11,part:7,pip:4,pixel:11,plot:2,point:3,practic:2,precompos:0,python:[2,7],pytorch:[5,7,10],recap:10,releas:2,represent:11,resourc:5,result:6,run:2,set:2,setup:[6,9],sourc:4,tensor:5,test:2,toolkit:7,train:[9,10],transform:11,tutori:[2,3],unit:11,up:2,upgrad:4,us:[4,7],user:3,util:0,v0:1,valid:[0,6],view:2,virtual:2,visual:[8,10],weight:8,why:7,wrapup:10,your:2}})