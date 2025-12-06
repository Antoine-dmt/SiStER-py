% map_plastic_strain

figure(1)
fastscatter(xm(im>1)/1e3,ym(im>1)/1e3,ep(im>1),'markersize',2);
set(gcf,'color','white')
set(gca,'YDir','reverse')
axis equal
colorbar
%colormap jet
title('PLASTIC STRAIN')
xlabel('cross-axis distance (km)','fontsize',15)
ylabel('depth (km)','fontsize',15)
