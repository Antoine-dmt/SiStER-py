% map_strain_rate
figure
fastscatter(xm(im>1)/1e3,ym(im>1)/1e3,log10(epsIIm(im>1)),'markersize',2);
set(gcf,'color','white')
set(gca,'YDir','reverse')
axis equal
colorbar
title('SECOND INVARIANT OF STRAIN RATE (s^{-1})')
xlabel('cross-axis distance (km)','fontsize',15)
ylabel('depth (km)','fontsize',15)
