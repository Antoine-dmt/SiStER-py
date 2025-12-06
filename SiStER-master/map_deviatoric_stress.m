% map_deviatoric_stress
sIIm=sqrt(sxxm.^2+sxym.^2);
figure
fastscatter(xm(im>1)/1e3,ym(im>1)/1e3,sIIm(im>1),'markersize',2);
set(gcf,'color','white')
set(gca,'YDir','reverse')
axis equal
colorbar
colormap jet
title('SECOND INVARIANT OF DEVIATORIC STRESS (Pa)')
xlabel('cross-axis distance (km)','fontsize',15)
ylabel('depth (km)','fontsize',15)
