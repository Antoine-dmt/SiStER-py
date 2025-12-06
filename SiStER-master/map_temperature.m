% map_temperature

figure
fastscatter(xm(im>1)/1e3,ym(im>1)/1e3,Tm(im>1),'markersize',2);
set(gcf,'color','white')
set(gca,'YDir','reverse')
axis equal
colorbar
colormap jet
title('TEMPERATURE (ºC)')
xlabel('cross-axis distance (km)','fontsize',15)
ylabel('depth (km)','fontsize',15)
