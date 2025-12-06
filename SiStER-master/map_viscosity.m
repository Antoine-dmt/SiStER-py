% map_viscosity
figure
fastscatter(xm(im>1)/1e3,ym(im>1)/1e3,log10(etam(im>1)),'markersize',2);
set(gcf,'color','white')
set(gca,'YDir','reverse')
axis equal
colorbar
colormap jet
title('VISCOSITY (Pa.s)')
xlabel('cross-axis distance (km)','fontsize',15)
ylabel('depth (km)','fontsize',15)
